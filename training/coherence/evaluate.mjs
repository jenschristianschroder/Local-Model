// Offline evaluation only. Uses the unchanged browser inference kernel.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import vm from 'node:vm';

const root = new URL('../../', import.meta.url);
const html = readFileSync(new URL('index.html', root), 'utf8');
const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1].split('// MODEL END:')[0];
const baseline = JSON.parse(html.match(/<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const candidateText = readFileSync(new URL('candidate.json', import.meta.url), 'utf8');
const candidate = JSON.parse(candidateText);
const reference = JSON.parse(readFileSync(new URL('reference.json', import.meta.url), 'utf8'));
assert.equal(createHash('sha256').update(candidateText).digest('hex'), reference.sha256);

function engine(payload) {
  const context = vm.createContext({ atob, document: {
    documentElement: { outerHTML: html },
    head: { outerHTML: html.match(/<head>[\s\S]*?<\/head>/)[0] },
    body: { outerHTML: html.match(/<body>[\s\S]*?<\/body>/)[0] },
    getElementById: id => ({ textContent: JSON.stringify(id === 'model-data' ? payload : {}) }),
  } });
  vm.runInContext(source + '\nglobalThis.engine={tokenize,promptMemory,predict,advance,decode,randomGenerator,sample};', context);
  return context.engine;
}

const engines = { baseline: engine(baseline), candidate: engine(candidate) };
let maxError = 0;
for (const item of reference.cases) {
  const model = engines.candidate;
  const ids = Array.from(model.tokenize(item.prompt), token => token.id);
  assert.deepEqual([1, ...ids], item.ids);
  const result = model.predict(model.promptMemory(ids));
  for (const field of ['hidden', 'logits']) {
    result[field].forEach((value, index) => {
      maxError = Math.max(maxError, Math.abs(value - item[field][index]));
    });
  }
}
assert(maxError < 0.0002, `Browser/PyTorch parity error ${maxError}`);

const stories = JSON.parse(readFileSync(new URL('test.json', import.meta.url), 'utf8'));
const novelStories = [
  { actor: 'Mira', object: 'silver bell', paragraphs: ['Mira rang the silver bell, and every candle in the village began singing. A badger asked her to stop the singing before the baby woke up.'] },
  { actor: 'Tobin', object: 'green mitten', paragraphs: ['Tobin found a green mitten full of tiny stars. A goose needed those stars to light the path home through the snowy forest.'] },
  { actor: 'Nella', object: 'wooden spoon', paragraphs: ['Nella stirred the Christmas soup with a wooden spoon. The soup asked for swimming lessons, so a rabbit brought a very small towel.'] },
  { actor: 'Orin', object: 'striped scarf', paragraphs: ['Orin tied a striped scarf around a shivering clock. A mole warned him that the clock had stopped, and Christmas morning could not begin until it ticked again.'] },
];
const cases = [];
const selected = stories.filter((_, index) => index % 7 === 0).slice(0, 12)
  .map(story => ({ ...story, group: 'held-out combination' }));
selected.push(...novelStories.map(story => ({ ...story, group: 'new plot' })));
for (const story of selected) {
  const prompt = story.paragraphs[0];
  for (const temperature of [0, 0.55]) {
    for (const [name, model] of Object.entries(engines)) {
      const memory = model.promptMemory(Array.from(model.tokenize(prompt), token => token.id));
      const rng = model.randomGenerator(42);
      const tokens = [];
      for (let step = 0; step < 512; step++) {
        const prediction = model.predict(memory, { temperature });
        const token = model.sample(prediction.probabilities, rng);
        tokens.push(token);
        model.advance(memory, token);
      }
      const text = model.decode(tokens);
      const paragraphs = text.split(/\n\n/).map(p => p.trim()).filter(Boolean);
      cases.push({ model: name, group: story.group, temperature, prompt, actor: story.actor, object: story.object,
        actorInNextThreeParagraphs: paragraphs.slice(0, 3).map(p => p.includes(story.actor)),
        objectInNextThreeParagraphs: paragraphs.slice(0, 3).map(p => p.includes(story.object)),
        text });
    }
  }
}
writeFileSync(new URL('evaluation.json', import.meta.url), JSON.stringify({
  checkpointSha256: reference.sha256, browserParityMaxError: maxError,
  caveat: 'Entity mentions are diagnostics, not a coherence score. Test combinations are held out; plot templates are shared. Baseline has a different vocabulary and training distribution.',
  cases,
}, null, 2) + '\n');
console.log(JSON.stringify({ browserParityMaxError: maxError, cases: cases.length }));
