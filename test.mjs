import vm from 'node:vm';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { verify } from './build-model.mjs';
const {html,checkpoint:payload,report,reference}=verify();
const source=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const numericalSource=source.split('// MODEL END:')[0];
const kernelSource=source.split('// INFERENCE KERNEL START')[1].split('// INFERENCE KERNEL END')[0];
function loadModel(edits={}){
  const context=vm.createContext({atob,document:{documentElement:{outerHTML:html},getElementById:id=>({textContent:JSON.stringify(id==='model-data'?payload:edits)})}});
  vm.runInContext(numericalSource+'\nglobalThis.m={bundle,model,loadWeights,modelStep,modelLogits,softmax,predict,forward,promptMemory,initialMemory,advance,tokenize,decode,randomGenerator,sample,weightAt,setWeight,applyOverrides};',context);
  return context.m;
}
const m=loadModel(),V=payload.architecture.vocabulary,H=payload.architecture.hidden;
const ids=text=>Array.from(m.tokenize(text),t=>t.id);
function generate(prompt,length=128,seed=42,temperature=.65,engine=m){
  const memory=engine.promptMemory(ids(prompt)),random=engine.randomGenerator(seed),output=[];
  for(let i=0;i<length;i++){
    const result=engine.predict(memory,{temperature});
    assert(result.logits.every(Number.isFinite));assert(Math.abs(result.probabilities.reduce((a,b)=>a+b,0)-1)<1e-10);
    const token=engine.sample(result.probabilities,random);output.push(token);engine.advance(memory,token);
  }
  return{output,memory,text:engine.decode(output)};
}
test('page is below 1 MB, parses, and has no external runtime requests',()=>{
  new vm.Script(source);assert(Buffer.byteLength(html)<1_000_000);
  assert(!/<script[^>]+src=/i.test(html));assert(!/\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(/.test(source));assert(!/@import|url\(https?:/i.test(html));
  const referenced=[...source.matchAll(/\$\('([^']+)'\)/g)].map(x=>x[1]);for(const id of referenced)assert(html.includes('id="'+id+'"'),'Missing DOM element: '+id);
});
test('all decoded tensors are finite and the learned tokenizer has valid merges',()=>{
  for(const[name,tensor]of Object.entries(payload.tensors)){const count=tensor.shape.reduce((a,b)=>a*b,1);assert.equal(m.model[name].length,count);assert(m.model[name].every(Number.isFinite));}
  for(const[a,b,out]of payload.tokenizer.merges){assert([a,b,out].every(id=>Number.isInteger(id)&&id>=0&&id<V));assert.equal(payload.tokenizer.vocabulary[a]+payload.tokenizer.vocabulary[b],payload.tokenizer.vocabulary[out]);}
});
test('browser BPE, hidden states, and logits match independent PyTorch reference vectors',()=>{
  for(const item of reference.cases){
    assert.deepEqual([1,...ids(item.prompt)],item.ids,item.prompt+' tokenization');
    const actual=m.forward(ids(item.prompt),{temperature:1});
    actual.hidden.forEach((v,i)=>assert(Math.abs(v-item.hidden[i])<.0001,item.prompt+' hidden '+i));
    actual.logits.forEach((v,i)=>assert(Math.abs(v-item.logits[i])<.0002,item.prompt+' logit '+i));
  }
});
test('numeric kernel runs without tokenizer, word labels, or DOM',()=>{
  const context=vm.createContext({});vm.runInContext(kernelSource+';globalThis.engine={modelStep,modelLogits,softmax};',context);
  let hidden=context.engine.modelStep(m.model,new Float32Array(H),1);
  for(const id of ids('The quiet forest'))hidden=context.engine.modelStep(m.model,hidden,id);
  assert.deepEqual(Array.from(context.engine.modelLogits(m.model,hidden)),Array.from(m.forward(ids('The quiet forest')).logits));
});
test('all vocabulary probabilities are exactly softmax of the raw logits',()=>{
  for(const prompt of ['','The robot hums','The robots hum','A curious cat','The distant planet','Strange 🦊'])for(const temperature of [.25,.65,1,1.5]){
    const prediction=m.forward(ids(prompt),{temperature}),maximum=Math.max(...prediction.logits);
    const exp=Array.from(prediction.logits,v=>Math.exp((v-maximum)/temperature)),sum=exp.reduce((a,b)=>a+b,0);
    prediction.probabilities.forEach((p,i)=>assert(Math.abs(p-exp[i]/sum)<1e-14));
    if(temperature>=1)assert(prediction.probabilities.every(p=>p>0),'No token is masked, including special tokens');
  }
});
test('zero output tensors yield a uniform distribution over every token',()=>{
  const isolated=loadModel();isolated.model.output.fill(0);isolated.model.outputBias.fill(0);
  assert(isolated.forward(ids('The little robot')).probabilities.every(p=>Math.abs(p-1/V)<1e-14));
});
test('weight edits directly change logits and survive model reload',()=>{
  const isolated=loadModel(),prompt=ids('The little robot'),before=isolated.forward(prompt,{temperature:1});
  let neuron=0;for(let h=1;h<H;h++)if(Math.abs(before.hidden[h])>Math.abs(before.hidden[neuron]))neuron=h;
  const token=0,old=isolated.weightAt(token,neuron),value=before.hidden[neuron]>=0?16:-16;
  isolated.setWeight(token,neuron,value);const after=isolated.forward(prompt,{temperature:1});
  assert(Math.abs(after.logits[token]-before.logits[token]-(value-old)*before.hidden[neuron])<.0001);
  assert(after.probabilities[token]>before.probabilities[token]);
  const reloaded=loadModel({version:4,weights:{[token+':'+neuron]:value}});assert.equal(reloaded.weightAt(token,neuron),value);
  assert.equal(loadModel({version:3,weights:{[token+':'+neuron]:value}}).weightAt(token,neuron),old);
});
test('sampler accepts special tokens and does not reject improbable outputs',()=>{
  const p=m.forward(ids('The robot hums'),{temperature:1}).probabilities;assert(p[0]>0);assert.equal(m.sample(p,()=>0),0);
  const special=new Float64Array(V);special[1]=1;assert.equal(m.sample(special,m.randomGenerator(42)),1);
});
test('novel English words, capitalization, and punctuation round-trip through word pieces',()=>{
  for(const prompt of ['The robotic telescope sparkles.','"Hello," said Lily.','A blue-green butterfly!','Unknownword123'])assert.equal(m.decode(ids(prompt)),prompt);
  assert.notDeepEqual(ids('The robot'),ids('the robot'));
});
test('seeded generation is reproducible and Continue preserves the same recurrent state',()=>{
  const a=generate('A curious cat',128,42),b=generate('A curious cat',128,42),c=generate('A curious cat',128,43);
  assert.deepEqual(a.output,b.output);assert.notDeepEqual(a.output,c.output);
  const memory=m.promptMemory(ids('A curious cat')),random=m.randomGenerator(42),split=[];
  for(const length of [64,64])for(let i=0;i<length;i++){const next=m.sample(m.predict(memory,{temperature:.65}).probabilities,random);split.push(next);m.advance(memory,next);}
  assert.deepEqual(split,a.output);assert.deepEqual(Array.from(memory.hidden),Array.from(a.memory.hidden));
});
test('1024 generated tokens stay finite with bounded recurrent state',()=>{
  const result=generate('The distant planet',1024);assert.equal(result.output.length,1024);assert(result.memory.hidden.every(v=>Number.isFinite(v)&&Math.abs(v)<=1));assert.equal(result.memory.recent.length,8);
});
test('training evidence and exported quantization metrics agree with the checkpoint',()=>{
  assert.equal(report.parameters,636908);assert(report.bestStep>=8000);assert(report.initialValidationLoss>7);assert(report.export.quantizedValidationLoss<3.3);
  assert(Math.abs(report.export.quantizedValidationLoss-report.export.floatValidationLoss)<.02);
  assert.equal(payload.training.quantizedValidationLoss,report.export.quantizedValidationLoss);
  if(report.extraCorpus)for(const split of ['train','valid']){const info=report.extraCorpus[split];const text=readFileSync(new URL('./training/'+info.file,import.meta.url));assert.equal(createHash('sha256').update(text).digest('hex'),info.sha256);}
});
