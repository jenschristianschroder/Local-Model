import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {JSDOM,VirtualConsole} from 'jsdom';
const html=readFileSync(new URL('./index.html',import.meta.url),'utf8');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(check){for(let i=0;i<100;i++){if(check())return;await delay(10);}assert(check(),'UI did not reach the expected state');}
async function app(saved,url='https://pocket.test/'){
  const errors=[],console=new VirtualConsole();console.on('jsdomError',e=>errors.push(e.message));
  const dom=new JSDOM(html,{url,runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:console,beforeParse(w){
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
test('copy story includes the full prompt and continuation, with fallback and honest failure feedback',async()=>{
  const a=await app();try{
    assert.equal(a.q('#copy-story').disabled,true);
    await a.go('predictions');a.click('#step');await delay(30);await a.go('response');
    const expected=a.q('#story-prompt').textContent+a.q('#story-continuation').textContent;
    a.click('[data-open="story-tools"]');let copied;
    Object.defineProperty(a.w.navigator,'clipboard',{configurable:true,value:{writeText:async text=>{copied=text;}}});
    a.click('#copy-story');await delay(10);assert.equal(copied,expected);assert.equal(a.q('#toast').textContent,'Story copied to clipboard.');
    a.w.navigator.clipboard.writeText=async()=>{throw Error('Permission denied');};
    a.w.document.execCommand=command=>{assert.equal(command,'copy');assert.equal(a.w.document.activeElement.value,expected);return true;};
    a.click('#copy-story');await delay(10);assert.equal(a.q('#toast').textContent,'Story copied to clipboard.');assert.equal(a.w.document.activeElement.id,'copy-story');
    a.w.document.execCommand=()=>false;a.click('#copy-story');await delay(10);
    assert.equal(a.q('#toast').textContent,'Copy unavailable. Use Save story as text.');assert.equal(a.q('#detail textarea'),null);assert.deepEqual(a.errors,[]);
  }finally{a.dom.window.close();}
});
test('example carousel preserves supplied text and supports expansion, buttons, keys and isolated swipes',async()=>{
  const a=await app();try{
    const stories=JSON.parse(readFileSync(new URL('./ui/examples.json',import.meta.url),'utf8'));
    const more=a.q('#samples [aria-controls]'),text=()=>a.w.document.getElementById(more.getAttribute('aria-controls')).textContent;
    const next=a.q('[aria-label="Next example story"]'),previous=a.q('[aria-label="Previous example story"]');
    for(let i=0;i<3;i++){
      assert.equal(a.q('#samples [role="status"]').textContent,(i+1)+' of 3');assert.equal(more.getAttribute('aria-expanded'),'false');
      more.click();assert.equal(text(),stories[i][1]);assert.equal(more.getAttribute('aria-expanded'),'true');next.click();
    }
    previous.click();assert.equal(a.q('#samples h3').textContent,stories[2][0]);
    previous.dispatchEvent(new a.w.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));assert.equal(a.q('#samples h3').textContent,stories[0][0]);
    const touch=(type,point)=>{const e=new a.w.Event(type,{bubbles:true});Object.defineProperty(e,type==='touchstart'?'touches':'changedTouches',{value:[point]});a.q('#samples h3').dispatchEvent(e);};
    touch('touchstart',{clientX:250,clientY:100});touch('touchend',{clientX:80,clientY:105});
    assert.equal(a.q('#samples h3').textContent,stories[1][0]);assert.equal(a.q('.page.is-active').id,'intro');
    touch('touchstart',{clientX:250,clientY:100});touch('touchend',{clientX:240,clientY:250});
    assert.equal(a.q('#samples h3').textContent,stories[1][0]);assert.deepEqual(a.errors,[]);
  }finally{a.dom.window.close();}
});
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
test('downloaded file remains usable when the file origin restricts history and storage',async()=>{
  const a=await app(undefined,'file:///pocket-lm.html');try{
    await a.go('predictions');const before=a.q('#prediction-context').textContent;
    a.click('#step');await delay(40);assert.notEqual(a.q('#prediction-context').textContent,before);
    a.click('[data-open="tokens"]');a.click('#close-sheet');await until(()=>!a.q('#detail').open);
    await a.go('inside');assert.equal(a.q('.page.is-active').id,'inside');assert.deepEqual(a.errors,[]);
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
