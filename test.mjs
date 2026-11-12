import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { test } from 'node:test';

const html=readFileSync(new URL('./index.html',import.meta.url),'utf8');
const source=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const numericalSource=source.split('// MODEL END:')[0];
const payload=JSON.parse(html.match(/<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const kernelSource=source.split('// INFERENCE KERNEL START')[1].split('// INFERENCE KERNEL END')[0];
function loadModel(edits={}){
  const context=vm.createContext({document:{documentElement:{outerHTML:html},getElementById:id=>({textContent:JSON.stringify(id==='model-data'?payload:edits)})}});
  vm.runInContext(numericalSource+'\nglobalThis.m={model,labels,modelStep,modelLogits,softmax,predict,forward,promptMemory,initialMemory,advance,tokenize,randomGenerator,sample,formatted,weightAt,setWeight,applyOverrides};',context);
  return context.m;
}
const m=loadModel(),{vocabulary,features,offsets:F,states:ST}=m.labels;
const id=word=>vocabulary.indexOf(word),ids=text=>m.tokenize(text).map(t=>t.id);
function generation(prompt,length=128,seed=42,options={}){
  const memory=m.promptMemory(ids(prompt)),random=m.randomGenerator(seed),output=[];
  for(let i=0;i<length;i++){
    const result=m.predict(memory,options),token=m.sample(result.probabilities,random);
    assert(result.logits.every(Number.isFinite));
    assert(Math.abs(result.probabilities.reduce((a,b)=>a+b,0)-1)<1e-10);
    assert(token>=0&&token<vocabulary.length);
    output.push(token);m.advance(memory,token);
  }
  return{memory,output,text:m.formatted(output.map(v=>vocabulary[v]))};
}
function probability(prompt,word,temperature=1){return m.forward(ids(prompt),{temperature}).probabilities[id(word)];}

test('the standalone application parses and makes no model or asset requests',()=>{
  new vm.Script(source);
  assert(!/<script[^>]+src=/i.test(html));
  assert(!/\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(/.test(source));
  assert(!/@import|url\(https?:/i.test(html));
});
test('the checkpoint contains only numeric inference data with valid sparse connections',()=>{
  const w=m.model,V=vocabulary.length,base=w.width+w.embeddingWidth;
  assert.equal(w.format,3);assert.equal(w.embedding.length,V);assert.equal(w.output.length,V);
  assert.equal(w.initial.length,w.width);assert.equal(w.readout.length,w.width);assert.equal(features.length,w.width);
  assert.equal(w.cellBias.length,w.cell.length);assert.equal(w.outputBias.length,V);
  function rowCheck(row,width){assert.equal(row.length%2,0);for(let j=0;j<row.length;j+=2){assert(Number.isInteger(row[j])&&row[j]>=0&&row[j]<width);assert(Number.isFinite(row[j+1]));}}
  w.embedding.forEach(row=>rowCheck(row,w.embeddingWidth));
  w.cell.forEach((row,n)=>rowCheck(row,base+n));
  w.output.forEach(row=>rowCheck(row,w.width));
  assert(w.readout.every(n=>Number.isInteger(n)&&n>=0&&n<base+w.cell.length));
  assert([...w.cellBias,...w.initial,...w.outputBias].every(Number.isFinite));
  const noStrings=value=>typeof value==='number'||(Array.isArray(value)?value.every(noStrings):Object.values(value).every(noStrings));
  assert(noStrings(w));
});
test('numeric-only kernel reproduces predictions without vocabulary labels, POS tables, or a DOM',()=>{
  const context=vm.createContext({});
  vm.runInContext(kernelSource+';globalThis.engine={modelStep,modelLogits,softmax};',context);
  const weights=JSON.parse(JSON.stringify(m.model));
  let hidden=Float64Array.from(weights.initial);
  for(const token of ids('the little robot hums'))hidden=context.engine.modelStep(weights,hidden,token);
  const actual=context.engine.softmax(context.engine.modelLogits(weights,hidden),.8);
  assert.deepEqual(Array.from(actual),Array.from(m.forward(ids('the little robot hums')).probabilities));
});
test('sampling probabilities are exactly softmax of raw model logits',()=>{
  for(const prompt of ['', 'the little robot hums','the robots hum','a','an','unknown 🦊','.','hums'])for(const temperature of [.05,.8,1,1.5]){
    const memory=m.promptMemory(ids(prompt)),result=m.predict(memory,{temperature});
    const logits=m.modelLogits(m.model,memory.hidden),maximum=Math.max(...logits);
    const e=Array.from(logits,x=>Math.exp((x-maximum)/temperature)),sum=e.reduce((a,b)=>a+b,0);
    assert.deepEqual(Array.from(result.logits),Array.from(logits));
    result.probabilities.forEach((p,v)=>assert(Math.abs(p-e[v]/sum)<1e-14));
    assert(Math.abs(result.probabilities.reduce((a,b)=>a+b,0)-1)<1e-10);
    if(temperature>=1)assert(result.probabilities.every(p=>p>0),'No vocabulary IDs are filtered');
  }
});
test('zero output weights and biases produce a uniform distribution over the entire vocabulary',()=>{
  const isolated=loadModel();isolated.model.output=isolated.model.output.map(()=>[]);isolated.model.outputBias.fill(0);
  const result=isolated.forward(ids('the little robot hums'),{temperature:1});
  assert(result.probabilities.every(p=>Math.abs(p-1/vocabulary.length)<1e-14));
});
test('the sampler does not skip special tokens or reject an unlikely choice',()=>{
  const p=m.forward(ids('the robot hums'),{temperature:1}).probabilities;
  assert(p[0]>0);assert.equal(m.sample(p,()=>0),0);
});
test('all four reported singular contexts strongly disfavor together through finite logits',()=>{
  for(const prompt of ['the little robot hums','the quiet forest blooms','a curious cat wakes','the distant planet sparkles','it can hum','hums'])for(const temperature of [.8,1,1.5]){
    const result=m.forward(ids(prompt),{temperature}),p=result.probabilities[id('together')];
    assert(Number.isFinite(result.logits[id('together')]));assert(p>0&&p<1e-6,prompt);
  }
});
test('plural subject activations make together a normal candidate',()=>{
  for(const prompt of ['the robots hum','the flowers bloom','the cats wake','the planets sparkle','they can hum','we work'])assert(probability(prompt,'together')>.02,prompt);
});
test('changing one output weight can make the original singular-subject error win',()=>{
  const v=id('together'),original=m.weightAt(v,F.alone),memory=m.promptMemory(ids('the little robot hums'));
  try{
    assert(m.predict(memory).probabilities[v]<1e-10);
    m.setWeight(v,F.alone,16);
    const result=m.predict(memory);
    assert(result.probabilities[v]>.999);
    assert.equal(m.sample(result.probabilities,m.randomGenerator(42)),v);
  }finally{m.setWeight(v,F.alone,original);}
});
test('verb affinities affect logits, and editing those weights can reverse the preference',()=>{
  const cases=[['the robot hums','softly','brightly'],['the forest blooms','slowly','patiently'],['the cat wakes','slowly','brightly'],['the planet sparkles','brightly','patiently'],['we wait','patiently','brightly'],['the river flows','gently','patiently']];
  for(const[prompt,good,odd]of cases)assert(probability(prompt,good)>probability(prompt,odd)*1e8,prompt);
  const v=id('brightly'),h=F.mismatch+m.labels.adverbs.indexOf('brightly'),original=m.weightAt(v,h);
  try{m.setWeight(v,h,16);assert(probability('the robot hums','brightly')>.999);}finally{m.setWeight(v,h,original);}
});
test('subject number lives in recurrent activations across objects, modals, and clauses',()=>{
  const cases=[['the robot sees birds',1,0],['the birds see a robot',0,1],['the robot dreams',1,0],['the dreams glow',0,1],['the birds glow . it can hum',1,0],['the robot glows . they can hum',0,1],['the robots hum and the cat wakes',1,0],['the robots hum .',0,0]];
  for(const[prompt,singular,plural]of cases){const h=m.promptMemory(ids(prompt)).hidden;assert.equal(h[F.singular],singular,prompt);assert.equal(h[F.plural],plural,prompt);}
});
test('syntax preferences come from weights rather than a support mask',()=>{
  assert(probability('the robot','builds')>probability('the robot','build')*1e8);
  assert(probability('the robots','build')>probability('the robots','builds')*1e8);
  const v=id('build'),original=m.weightAt(v,ST.singular);
  try{m.setWeight(v,ST.singular,64);assert(probability('the robot','build')>.999);}finally{m.setWeight(v,ST.singular,original);}
});
test('swapping numeric token embeddings changes number behavior without changing word labels',()=>{
  const isolated=loadModel();isolated.model.embedding[id('robots')]=isolated.model.embedding[id('robot')].slice();
  assert.deepEqual(Array.from(isolated.forward(ids('the robots hums')).logits),Array.from(isolated.forward(ids('the robot hums')).logits));
});
test('ambiguous and out-of-order input stays bounded inside the recurrent cell',()=>{
  const prompts=['the secret forest','the secret secret story','the dream dreams','unknown 🦊 the secret','hums blooms robot secret .'];
  for(const prompt of prompts){
    const h=m.promptMemory(ids(prompt)).hidden;
    assert(h.every(Number.isFinite));assert.equal(h.slice(0,m.labels.stateNames.length).reduce((a,b)=>a+b,0),1,prompt);
  }
  const memory=m.initialMemory(),random=m.randomGenerator(9);
  for(let i=0;i<2000;i++)m.advance(memory,Math.floor(random()*vocabulary.length));
  assert(memory.hidden.every(x=>Number.isFinite(x)&&Math.abs(x)<=64));
  assert.equal(memory.state.reduce((a,b)=>a+b,0),1);
});
test('article and adjective preferences are finite model scores',()=>{
  const a=m.forward(ids('a'),{temperature:1}),an=m.forward(ids('an'),{temperature:1});
  assert(a.probabilities[id('little')]>a.probabilities[id('electric')]*1e6);
  assert(an.probabilities[id('electric')]>an.probabilities[id('little')]*1e6);
  assert(a.probabilities[id('electric')]>0);
  const before=probability('the little','curious'),after=probability('the little silver','curious');
  assert(after<before*1e-5);
});
test('repetition feedback is encoded in recurrent token activations and output weights',()=>{
  const memory=m.promptMemory(ids('the robot sees a robot . the')),v=id('robot'),h=F.trace+v,original=m.weightAt(v,h);
  assert(memory.hidden[h]>.3);assert(original<0);
  const before=m.predict(memory,{temperature:1}).probabilities[v];
  try{m.setWeight(v,h,0);assert(m.predict(memory,{temperature:1}).probabilities[v]>before);}finally{m.setWeight(v,h,original);}
});
test('display history cannot influence the model prediction',()=>{
  const memory=m.promptMemory(ids('the little robot hums')),before=Array.from(m.predict(memory).logits);
  memory.recent=ids('they glow together and they glow together');
  assert.deepEqual(Array.from(m.predict(memory).logits),before);
});
test('512-token generation remains finite and crosses sentence and paragraph boundaries',()=>{
  for(const prompt of ['the little robot','the quiet forest','a curious cat','the distant planet','we']){
    const result=generation(prompt,512);
    assert.equal(result.output.length,512);assert(result.text.includes('\n\n'));
    assert(result.output.filter(v=>vocabulary[v]==='.').length>10);
    assert(result.memory.hidden.every(Number.isFinite));assert(result.memory.recent.length<=8);
  }
});
test('continuation uses the same recurrent state and random sequence',()=>{
  const memory=m.promptMemory(ids('the little robot')),rng=m.randomGenerator(42),output=[];
  for(const batch of [64,64])for(let i=0;i<batch;i++){const token=m.sample(m.predict(memory).probabilities,rng);output.push(token);m.advance(memory,token);}
  assert.deepEqual(output,generation('the little robot',128).output);
});
test('greedy and high-temperature generation remain numerically stable',()=>{
  for(const temperature of [0,1.5])assert(generation('the little robot',512,7,{temperature}).memory.hidden.every(Number.isFinite));
});
test('temperature scales the full vocabulary distribution and zero selects the maximum',()=>{
  const memory=m.promptMemory(ids('the little robot'));
  const entropy=p=>p.reduce((sum,x)=>sum-(x>0?x*Math.log(x):0),0);
  assert(entropy(m.predict(memory,{temperature:1.5}).probabilities)>entropy(m.predict(memory,{temperature:.1}).probabilities));
  const result=m.predict(memory,{temperature:0});assert.equal(result.probabilities[result.logits.indexOf(Math.max(...result.logits))],1);
});
test('edited weights survive JSON checkpoints and versioned standalone HTML overrides',()=>{
  const v=id('together'),h=F.alone;
  for(const value of [-48,64])assert.equal(loadModel({version:3,weights:{[v+':'+h]:value}}).weightAt(v,h),value);
  assert.equal(loadModel({version:2,weights:{[v+':'+h]:64}}).weightAt(v,h),-24);
  const isolated=loadModel({version:3,weights:{[v+':'+h]:16}}),weights=JSON.parse(JSON.stringify(isolated.model));
  const memory=isolated.promptMemory(ids('the little robot hums'));
  assert.deepEqual(Array.from(isolated.modelLogits(weights,memory.hidden)),Array.from(isolated.predict(memory).logits));
});
test('seeded runs reproduce the same samples while different seeds change the text',()=>{
  assert.equal(generation('the little robot',64,42).text,generation('the little robot',64,42).text);
  assert.notEqual(generation('the little robot',64,42).text,generation('the little robot',64,18).text);
});
console.log(JSON.stringify(m.labels.stats));
