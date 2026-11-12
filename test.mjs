import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { test } from 'node:test';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const numericalSource = source.split('// MODEL END:')[0];
function loadModel(edits={}) {
  const context = vm.createContext({document:{documentElement:{outerHTML:html},getElementById:()=>({textContent:JSON.stringify(edits)})}});
  vm.runInContext(numericalSource+'\nglobalThis.model={vocabulary,embeddings,features,A,B,bg,W2,b2,support,ST,stateNames,TOPICS,SEM,forward,predict,promptMemory,initialMemory,advance,tokenize,randomGenerator,sample,formatted,parameterCount,V,D,H,G,S,BOS,UNK,has};',context);
  return context.model;
}
const m=loadModel();
const ids=text=>m.tokenize(text).map(t=>t.id);
const id=word=>m.vocabulary.indexOf(word);
function generate(prompt,length=128,seed=42,options={}) {
  const memory=m.promptMemory(ids(prompt)),random=m.randomGenerator(seed),output=[];
  for(let i=0;i<length;i++) {
    const prediction=m.predict(memory,options);
    const token=m.sample(prediction.probabilities,random);
    assert.equal(prediction.allowed[token],1);
    assert(token!==m.BOS&&token!==m.UNK);
    assert(Math.abs(prediction.probabilities.reduce((a,b)=>a+b,0)-1)<1e-10);
    output.push(token);m.advance(memory,token);
  }
  return {output,memory,text:m.formatted(output.map(i=>m.vocabulary[i]))};
}
test('single-file application parses without dependencies or network requests',()=>{
  new vm.Script(source);
  assert(!/<script[^>]+src=/i.test(html));
  assert(!/\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(/.test(source));
  assert(!/@import|url\(https?:/i.test(html));
});
test('larger vocabulary is unique and all numeric matrix shapes agree',()=>{
  assert(m.V>400);assert(m.parameterCount>40000);
  assert.equal(new Set(m.vocabulary).size,m.V);
  assert(m.embeddings.every(row=>row.length===m.D));
  assert(m.A.every(row=>row.length===m.S+21));
  assert(m.B.every(row=>row.length===m.G));
  assert(m.W2.every(row=>row.length===m.H));
});
test('probabilities stay finite and normalized for normal and unknown prompts',()=>{
  for(const prompt of ['', 'the little robot','the quiet forest','we','unknown 🦊','.','a','an','the robot builds a small bridge'])
    for(const temperature of [0,.05,.8,1.5])for(const topP of [.1,.92,1]){
      const p=m.forward(ids(prompt),{temperature,topP}).probabilities;
      assert(p.every(n=>Number.isFinite(n)&&n>=0&&n<=1.0000000001));
      assert(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-10);
      assert.equal(p[m.BOS],0);assert.equal(p[m.UNK],0);
      if(temperature===0)assert.equal(p.filter(n=>n===1).length,1);
    }
});
test('generation crosses sentence boundaries and reaches 512 tokens',()=>{
  for(const prompt of ['the little robot','the quiet forest','the distant planet','a curious cat','we']){
    const result=generate(prompt,512);
    assert.equal(result.output.length,512);
    assert(result.output.filter(t=>m.has(t,'punct')).length>10);
    assert(result.text.includes('\n\n'));
    assert(result.memory.recent.length<=64);
    assert(result.memory.theme.every(n=>Number.isFinite(n)&&n>=0&&n<=1));
  }
});
test('incremental continuation matches uninterrupted generation',()=>{
  const memory=m.promptMemory(ids('the little robot')),random=m.randomGenerator(42),output=[];
  for(const batch of [64,64])for(let i=0;i<batch;i++){const token=m.sample(m.predict(memory).probabilities,random);output.push(token);m.advance(memory,token);}
  assert.deepEqual(output,generate('the little robot',128).output);
});
test('state retains prompt theme beyond the former four-token context',()=>{
  const memory=m.promptMemory(ids('the robot'));
  for(const token of ids('is quiet . it is gentle . it'))m.advance(memory,token);
  assert(memory.anchor[0]>.9);
  assert(memory.subject[m.SEM.indexOf('machine')]===1);
  assert(memory.state[m.ST.singular]===1);
});
test('recurrent grammar resolves ambiguous words and subject number',()=>{
  assert.equal(m.promptMemory(ids('the robot dreams')).state[m.ST.verbDone],1);
  assert.equal(m.promptMemory(ids('the dreams')).state[m.ST.plural],1);
  const singular=m.forward(ids('the robot')).probabilities,plural=m.forward(ids('the robots')).probabilities;
  assert.equal(singular[id('build')],0);assert.equal(plural[id('builds')],0);
});
test('article constraints and adjective cap are applied before sampling',()=>{
  assert.equal(m.forward(ids('a')).probabilities[id('electric')],0);
  assert.equal(m.forward(ids('an')).probabilities[id('little')],0);
  assert.equal(m.forward(ids('the little silver')).probabilities[id('curious')],0);
});
test('lower top-p narrows the candidate pool and temperature raises entropy',()=>{
  const memory=m.promptMemory(ids('the little robot'));
  assert(m.predict(memory,{topP:.1}).probabilities.filter(x=>x>0).length<m.predict(memory,{topP:1}).probabilities.filter(x=>x>0).length);
  const entropy=p=>p.reduce((s,x)=>s-(x>0?x*Math.log(x):0),0);
  assert(entropy(m.predict(memory,{temperature:1.5,topP:1}).probabilities)>entropy(m.predict(memory,{temperature:.1,topP:1}).probabilities));
});
test('repetition penalty reduces a recently used content word',()=>{
  const memory=m.promptMemory(ids('the robot sees a robot . the'));
  const base=m.predict(memory,{repetition:0,noRepeat:false,topP:1}).probabilities[id('robot')];
  assert(m.predict(memory,{repetition:2,noRepeat:false,topP:1}).probabilities[id('robot')]<base);
});
test('repeated-trigram guard masks a previously used phrase',()=>{
  const memory=m.promptMemory(ids('the little robot glows . the little'));
  assert(m.predict(memory,{noRepeat:false,topP:1}).probabilities[id('robot')]>0);
  assert.equal(m.predict(memory,{noRepeat:true,topP:1}).probabilities[id('robot')],0);
});
test('active output weights change probabilities and exports reload all slider values',()=>{
  const word=id('dreams'),feature=m.ST.singular,memory=m.promptMemory(ids('the little robot'));
  const before=m.predict(memory,{topP:1}).probabilities[word],original=m.W2[word][feature];
  m.W2[word][feature]+=5;assert(m.predict(memory,{topP:1}).probabilities[word]>before);m.W2[word][feature]=original;
  for(const value of [-16,24])assert.equal(loadModel({version:2,weights:{[word+':'+feature]:value}}).W2[word][feature],value);
  assert.equal(loadModel({version:1,weights:{[word+':'+feature]:24}}).W2[word][feature],original);
  assert.equal(loadModel({version:2,weights:{[word+':'+feature]:999}}).W2[word][feature],original);
});
test('seeded generation is repeatable but different seeds change the result',()=>{
  assert.equal(generate('the little robot',64,42).text,generate('the little robot',64,42).text);
  assert.notEqual(generate('the little robot',64,42).text,generate('the little robot',64,18).text);
});
test('long greedy and high-temperature runs remain bounded and valid',()=>{
  for(const temperature of [0,1.5]){
    const result=generate('the little robot',512,7,{temperature});
    assert.equal(result.output.length,512);
    assert.equal(result.memory.recent.length,64);
  }
});
console.log(JSON.stringify({vocabulary:m.V,parameters:m.parameterCount,reluGates:m.G,recurrentStates:m.S,outputFeatures:m.H}));
console.log('Example: '+generate('the little robot',80).text);
