import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {JSDOM,VirtualConsole} from 'jsdom';
const html=readFileSync(new URL('./index.html',import.meta.url),'utf8');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(check){for(let i=0;i<100;i++){if(check())return;await delay(10);}assert(check(),'UI did not reach the expected state');}
async function app(saved){
  const errors=[],console=new VirtualConsole();console.on('jsdomError',e=>errors.push(e.message));
  const dom=new JSDOM(html,{url:'https://pocket.test/',runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:console,beforeParse(w){
    w.TextEncoder=TextEncoder;
    // jsdom has no top-layer layout; native dialog behavior is checked in-browser.
    w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
    w.HTMLDialogElement.prototype.close=function(){this.open=false;};
    if(saved)w.sessionStorage.setItem('pocket-lm-session-v1',saved);
  }});
  await delay(30);assert.deepEqual(errors,[]);
  const w=dom.window,q=s=>w.document.querySelector(s),click=s=>{assert(q(s),s);q(s).click();};
  const go=async id=>{click('[data-open="chapters"]');const buttons=[...q('#sheet-chapters').querySelectorAll('button')];buttons[[...w.document.querySelectorAll('.page')].findIndex(p=>p.id===id)].click();await until(()=>q('.page.is-active').id===id&&!q('#detail').open);};
  return {dom,w,q,click,go,errors};
}
test('guided navigation preserves a paused story and details dismiss through browser Back',async()=>{
  const a=await app();try{
    await a.go('playground');a.click('[data-open="settings"]');
    a.q('#slow').checked=true;a.q('#limit').value='1024';a.click('#close-sheet');await until(()=>!a.q('#detail').open);
    a.click('#generate');await until(()=>a.q('#story-continuation').textContent.length>0);
    a.click('#next-page');await until(()=>a.q('.page.is-active').id==='predictions');
    const story=a.q('#story-continuation').textContent;await delay(150);assert.equal(a.q('#story-continuation').textContent,story);
    assert.equal(a.q('#status').dataset.running,'false');
    a.click('[data-open="tokens"]');assert.equal(a.q('#detail').open,true);assert.equal(a.w.document.activeElement.id,'sheet-title');
    a.w.history.back();await until(()=>!a.q('#detail').open);
    assert.equal(a.q('.page.is-active').id,'predictions');
    a.click('#step');await delay(30);assert.notEqual(a.q('#prediction-context').textContent.trim(),story.trim());
    await a.go('response');assert(a.q('#story-continuation').textContent.startsWith(story));assert.deepEqual(a.errors,[]);
  }finally{a.dom.window.close();}
});
test('weight experiment updates actual probability, resets, and retains an explored connection',async()=>{
  const a=await app();try{
    await a.go('weights');const before=a.q('#after-prob').textContent;
    for(let i=0;i<5;i++)a.click('#weight-up');await delay(30);
    assert.notEqual(a.q('#after-prob').textContent,before);assert.equal(a.q('#before-prob').textContent,before);
    a.click('#restore-connection');await delay(30);assert.equal(a.q('#after-prob').textContent,before);
    a.click('[data-open="network"]');const neuron=a.q('[data-neuron="5"]');neuron.focus();neuron.click();
    await delay(30);assert.equal(a.w.document.activeElement,neuron,'Selecting a neuron must not destroy its focused control');
    assert.equal(neuron.getAttribute('aria-pressed'),'true');
    a.click('#explorer-edit');await until(()=>!a.q('#detail').open);assert(a.q('#connection-name').textContent.startsWith('Neuron 6 →'));
    assert.deepEqual(a.errors,[]);
  }finally{a.dom.window.close();}
});
test('session restore resumes the exact seeded continuation including manual token choices',async()=>{
  const a=await app();let saved,expected;try{
    await a.go('predictions');a.click('#step');await delay(30);
    a.q('#probabilities button').focus();a.q('#probabilities button').click();await delay(30);
    assert.equal(a.w.document.activeElement,a.q('#probabilities button'),'Keyboard focus must stay on the refreshed candidate');
    a.click('#step');await delay(350);
    saved=a.w.sessionStorage.getItem('pocket-lm-session-v1');const snapshot=JSON.parse(saved);
    assert.equal(snapshot.generated.length,3);assert.equal(snapshot.generated[1].forced,true);
    const m=a.w.PocketLM,r=m.randomGenerator(snapshot.seed);for(const t of snapshot.generated)if(!t.forced)r();
    expected=m.sample(m.predict({hidden:new Float32Array(snapshot.hidden),recent:snapshot.recent},{temperature:snapshot.temperature}).probabilities,r);
  }finally{a.dom.window.close();}
  const b=await app(saved);try{
    assert.equal(b.q('.page.is-active').id,'predictions');b.click('#step');await delay(350);
    const restored=JSON.parse(b.w.sessionStorage.getItem('pocket-lm-session-v1'));
    assert.equal(restored.generated.length,4);assert.equal(restored.generated.at(-1).id,expected);assert.deepEqual(b.errors,[]);
  }finally{b.dom.window.close();}
});
