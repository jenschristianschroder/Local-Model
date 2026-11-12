import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { test } from 'node:test';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const numericalSource = source.split('const $=id=>')[0];
function loadModel(edits = {}) {
  const context = vm.createContext({ document: {
    documentElement: { outerHTML: html },
    getElementById: () => ({ textContent: JSON.stringify(edits) }),
  }});
  vm.runInContext(numericalSource + '\nglobalThis.model={vocabulary,embeddings,neurons,W1,b1,W2,b2,HN,forward,tokenize,randomGenerator,sample,V,D,C,H,BOS,UNK,END};', context);
  return context.model;
}
const m = loadModel();
const ids = prompt => m.tokenize(prompt).map(t => t.id);
function generate(prompt, seed=42, temperature=.75) {
  const context = ids(prompt), output = [], random = m.randomGenerator(seed);
  for(let i=0;i<40;i++) {
    const next = m.sample(m.forward(context, temperature).probabilities, random);
    if(next === m.END) break;
    context.push(next); output.push(m.vocabulary[next]);
  }
  return output.join(' ').replace(/ ([.!])/g,'$1');
}

test('the full inline JavaScript parses, with no runtime dependencies', () => {
  new vm.Script(source);
  assert.equal(/<script[^>]+src=/i.test(html), false);
  assert.equal(/\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(/.test(source), false);
  assert.equal(/@import|url\(https?:/i.test(html), false);
});
test('vocabulary is unique and all parameter shapes agree', () => {
  assert.equal(new Set(m.vocabulary).size, m.V);
  assert(m.embeddings.every(row => row.length === m.D));
  assert(m.W1.every(row => row.length === m.C*m.D));
  assert(m.W2.every(row => row.length === m.H));
  assert.equal(m.W2.length, m.V);
});
test('probabilities are finite, normalized, and exclude input-only tokens', () => {
  for(const prompt of ['', 'the little robot', 'the quiet forest', 'we', 'a curious cat', 'unknown 🦊', 'the robot sees a bright star', '.']) {
    for(const temp of [0,.05,.75,1.5]) {
      const p = m.forward(ids(prompt), temp).probabilities;
      assert(p.every(n => Number.isFinite(n) && n >= 0 && n <= 1));
      assert(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-12);
      assert.equal(p[m.BOS],0); assert.equal(p[m.UNK],0);
      if(temp===0) assert.equal(p.filter(n=>n===1).length,1);
    }
  }
});
test('inference uses exactly the last four tokens', () => {
  assert.deepEqual(Array.from(m.forward(ids('a cat sees the little silver robot')).probabilities), Array.from(m.forward(ids('the little silver robot')).probabilities));
});
test('increasing an active output weight raises that token probability', () => {
  const robot=ids('the little robot'), word=m.vocabulary.indexOf('dreams'), h=m.HN.singular;
  const before=m.forward(robot).probabilities[word], original=m.W2[word][h];
  m.W2[word][h]+=4;
  const after=m.forward(robot).probabilities[word];
  m.W2[word][h]=original;
  assert(after>before*2);
});
test('temperature raises entropy and fixed seeds reproduce generation', () => {
  const entropy = p => p.reduce((s,x)=>s-(x>0?x*Math.log(x):0),0);
  assert(entropy(m.forward(ids('the little robot'),1.5).probabilities)>entropy(m.forward(ids('the little robot'),.1).probabilities));
  assert.equal(generate('the little robot',42),generate('the little robot',42));
  assert.notEqual(generate('the little robot',42),generate('the little robot',18));
});
test('punctuation finishes generation and unknown words use UNK', () => {
  assert(m.forward(ids('.')).probabilities[m.END]>.9999);
  assert.equal(m.tokenize('unlistedword')[0].id,m.UNK);
});
test('downloaded model edits support the entire slider range', () => {
  const word=m.vocabulary.indexOf('dreams'), h=m.HN.singular;
  assert.equal(loadModel({[word+':'+h]:24}).W2[word][h],24);
  assert.equal(loadModel({[word+':'+h]:-16}).W2[word][h],-16);
  assert.equal(loadModel({[word+':'+h]:999}).W2[word][h],m.W2[word][h]);
});
for(const prompt of ['the little robot','the quiet forest','a curious cat','we']) console.log(prompt+' → '+generate(prompt));
console.log(JSON.stringify({vocabulary:m.V,embedding:m.D,hidden:m.H,parameters:m.V*m.D+m.H*m.C*m.D+m.H+m.V*m.H+m.V}));
