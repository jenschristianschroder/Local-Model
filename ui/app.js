const $=id=>document.getElementById(id),MAX_DOCUMENT=4096;
const pageList=Array.from(document.querySelectorAll('.page')),dialog=$('detail');
let activePage=0,ready=false,sheet='',sheetTrigger=null,closingHistory=false;
let promptTokens=[],generated=[],memory=initialMemory(),rng=randomGenerator(42),running=false,timer=null,runTarget=0,latest=null,ranking=null,frame=null;
let selectedH=0,selectedV=0,baseline=null,status='Ready when you are',toastTimer,swipe=null,sessionTimer;
const options=()=>({temperature:Number($('temperature').value)});
const allIds=()=>promptTokens.map(t=>t.id).concat(generated.map(t=>t.id));
const currentId=()=>pageList[activePage].id;
const number=n=>n.toLocaleString('en-US');
const probabilityText=p=>p>=.0001?(p*100).toFixed(2)+'%':p===0?'0%':(p*100).toExponential(1)+'%';
const signed=x=>(x>=0?'+':'')+x.toFixed(3);
function infer(){latest=predict(memory,options());ranking=null;baseline=null;}
function ranked(){return ranking??=Array.from(latest.probabilities,(p,id)=>({id,p})).sort((a,b)=>b.p-a.p);}
function node(tag,className,text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
function setText(id,text){$(id).textContent=text;}
function announce(text){setText('announcement',text);}
function stop(message='Paused · continue whenever you like'){
if(timer!==null)clearTimeout(timer);timer=null;running=false;status=message;scheduleSave();
}
function render(){if(frame===null)frame=requestAnimationFrame(()=>{frame=null;renderNow();});}
function renderNow(){
if(!latest)return;
const capped=generated.length>=MAX_DOCUMENT;
$('generate').disabled=capped;
$('continue-generation').disabled=capped&&!running;
setText('continue-generation',running?'Stop writing':generated.length?'Continue story':'Generate story');
$('step').disabled=capped||running;
$('save-text').disabled=!generated.length;
$('copy-story').disabled=!generated.length;
setText('status',running?'Writing on your device…':status);$('status').dataset.running=String(running);
if(currentId()==='response')renderOutput();
if(currentId()==='predictions')renderPredictions();
if(currentId()==='inside')renderNetwork();
if(currentId()==='weights')renderEditor();
if(sheet)renderSheet();
}
function reset(){
stop('Ready when you are');promptTokens=tokenize($('prompt').value);generated=[];
memory=promptMemory(promptTokens.map(t=>t.id));rng=randomGenerator(Number($('seed').value)||0);infer();
const unknown=promptTokens.filter(t=>t.id===UNK).length;
$('input-note').hidden=!unknown;setText('input-note',unknown+' unsupported character(s) use an unknown token.');
selectTop();render();scheduleSave();
}
function appendToken(forcedId){
if(generated.length>=MAX_DOCUMENT){stop('Document limit reached · start a new story');render();return;}
const id=forcedId===undefined?sample(latest.probabilities,rng):forcedId;
generated.push({id,forced:forcedId!==undefined});advance(memory,id);
if(running&&$('one-sentence').checked&&/[.!?]["']?$/.test(decode([id])))stop('Sentence complete · continue to add more');
else if(running&&generated.length>=runTarget)stop('Batch complete · continue to add more');
else if(!running)status=forcedId===undefined?'One token added':'You chose a token';
if(generated.length>=MAX_DOCUMENT)stop('Document limit reached · start a new story');
infer();render();
if(!running){announce((forcedId===undefined?'Generated ':'You chose ')+pieceLabel(id)+'. '+decode(allIds()).slice(-220));scheduleSave();}
}
function tick(){
if(!running)return;
try{appendToken();if(running)timer=setTimeout(tick,$('slow').checked?90:4);}
catch(error){stop('Generation paused');toast('Could not generate: '+error.message);render();}
}
function generateStory(){
if(running){stop();render();return;}
goToPage('response');runTarget=Math.min(MAX_DOCUMENT,generated.length+Number($('limit').value));running=true;render();tick();
}
function renderOutput(){
const scroller=$('story-scroll'),atEnd=scroller.scrollHeight-scroller.scrollTop-scroller.clientHeight<65;
const prefix=decode(promptTokens.map(t=>t.id));
setText('story-prompt',$('prompt').value);setText('story-continuation',decode(allIds()).slice(prefix.length));
$('story-placeholder').hidden=generated.length>0;
if(atEnd)scroller.scrollTop=scroller.scrollHeight;
}
function renderCandidates(target,count){
const focus=Array.from(target.children).indexOf(document.activeElement);
target.replaceChildren();
for(const t of ranked().slice(0,count)){
const b=node('button','prob');b.disabled=generated.length>=MAX_DOCUMENT||running;
b.setAttribute('aria-label','Choose '+pieceLabel(t.id)+', '+probabilityText(t.p));
const top=node('span','prob-top');top.append(node('span','',pieceLabel(t.id)),node('span','',probabilityText(t.p)));
const track=node('span','prob-track'),fill=node('span','prob-fill');fill.style.width=t.p*100+'%';track.append(fill);b.append(top,track);
b.addEventListener('click',()=>appendToken(t.id));target.append(b);
}
if(focus>=0)target.children[focus]?.focus({preventScroll:true});
}
function renderPredictions(){
const text=decode(allIds());setText('prediction-context',(text.length>100?'…':'')+text.slice(-100));
renderCandidates($('probabilities'),3);
}
function selectTop(){
selectedV=ranked()[0].id;selectedH=0;
for(let h=1;h<H;h++)if(Math.abs(weightAt(selectedV,h)*latest.hidden[h])>Math.abs(weightAt(selectedV,selectedH)*latest.hidden[selectedH]))selectedH=h;
}
function connectionName(){return features[selectedH]+' → '+pieceLabel(selectedV);}
function connectionMath(){return signed(latest.hidden[selectedH])+' × '+signed(weightAt(selectedV,selectedH))+' = '+signed(latest.hidden[selectedH]*weightAt(selectedV,selectedH));}
function renderNetwork(){
setText('network-input',pieceLabel(memory.recent.at(-1)??PARAGRAPH));setText('network-neuron',features[selectedH]);
setText('network-activation',signed(latest.hidden[selectedH]));$('network-activation').classList.toggle('negative',latest.hidden[selectedH]<0);
$('network-predictions').replaceChildren();
for(const t of ranked().slice(0,3)){const row=node('div','mini-row');row.append(node('span','',pieceLabel(t.id)),node('span','',probabilityText(t.p)));$('network-predictions').append(row);}
}
function renderEditor(){
setText('connection-name',connectionName());$('output-weight').value=weightAt(selectedV,selectedH);
$('output-weight').setAttribute('aria-label','Weight from '+features[selectedH]+' to '+pieceLabel(selectedV));
setText('weight-value',weightAt(selectedV,selectedH).toFixed(3));
baseline??=softmax(modelLogits({...model,output:originalWeights},memory.hidden),Number($('temperature').value));
const before=baseline[selectedV],after=latest.probabilities[selectedV];
setText('before-prob',probabilityText(before));setText('after-prob',probabilityText(after));
$('before-bar').style.width=before*100+'%';$('after-bar').style.width=after*100+'%';
setText('weight-help',Number($('temperature').value)===0?'At temperature zero, the highest-scoring token gets 100%. Use Generation settings to see softer probabilities.':latest.hidden[selectedH]>=0?'This activation is positive. A higher weight raises this token’s score.':'This activation is negative. A higher weight lowers this token’s score.');
$('restore-connection').disabled=weightAt(selectedV,selectedH)===originalWeights[selectedV*H+selectedH];
}
function changeWeight(value){
value=Math.max(-16,Math.min(16,value));stop('Connection updated');setWeight(selectedV,selectedH,value);
const key=selectedV+':'+selectedH;
if(weightAt(selectedV,selectedH)===originalWeights[selectedV*H+selectedH])delete overrides[key];else overrides[key]=weightAt(selectedV,selectedH);
infer();render();scheduleSave();
}
function writeHistory(mode,state){
try{history[mode+'State'](state,'','#'+currentId());}
catch{try{history[mode+'State'](state,'');}catch{/* Navigation still works if a file viewer denies history. */}}
}
function goToPage(target,{historyMode='push',focus=true,keepConnection=false}={}){
const index=typeof target==='number'?target:pageList.findIndex(p=>p.id===target);
if(index<0||index>=pageList.length)return;
const previous=currentId(),changed=index!==activePage;activePage=index;
if(ready&&running&&currentId()!=='response'){stop('Paused while you explore');announce('Story paused. Your place is saved.');}
pageList.forEach((p,i)=>{p.classList.toggle('is-active',i===index);p.inert=i!==index;p.setAttribute('aria-hidden',String(i!==index));});
setText('page-position',(index+1)+' of '+pageList.length);$('progress').style.width=(index+1)/pageList.length*100+'%';
$('previous-page').disabled=index===0;setText('next-prefix',index===pageList.length-1?'Keep exploring':'Up next');
setText('next-label',index===pageList.length-1?'Experiment again':pageList[index+1].dataset.title);
document.querySelectorAll('#chapter-list button').forEach((b,i)=>{if(i===index)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
if(historyMode==='push'&&(changed||location.hash!=='#'+currentId()))writeHistory('push',{slide:currentId()});
if(historyMode==='replace')writeHistory('replace',{slide:currentId()});
if(ready){
if(changed&&!keepConnection&&(currentId()==='inside'||(currentId()==='weights'&&previous!=='inside')))selectTop();
renderNow();scheduleSave();
}
if(focus&&changed)pageList[index].querySelector('h1,h2').focus({preventScroll:true});
}
function fillTokens(target,ids){target.replaceChildren();for(const id of ids)target.append(node('span','token-pill',pieceLabel(id)));}
function renderSheet(){
if(sheet==='connection'){
setText('detail-connection',connectionName());setText('connection-math',connectionMath());
const contribution=latest.hidden[selectedH]*weightAt(selectedV,selectedH);
setText('connection-impact','This connection '+(contribution>=0?'raises':'lowers')+' the score by '+Math.abs(contribution).toFixed(3)+'. The token’s current probability is '+probabilityText(latest.probabilities[selectedV])+'.');
}
if(sheet==='network')renderExplorer();
if(sheet==='tokens')fillTokens($('token-example'),promptTokens.slice(0,14).map(t=>t.id));
if(sheet==='predictions-more'){
renderCandidates($('more-probabilities'),12);setText('remaining',probabilityText(Math.max(0,1-ranked().slice(0,12).reduce((s,t)=>s+t.p,0)))+' across the other '+(V-12)+' tokens.');
setText('state-summary',H+' activations · state magnitude '+Math.sqrt(memory.hidden.reduce((s,v)=>s+v*v,0)).toFixed(3));fillTokens($('context'),memory.recent);
}
if(sheet==='vocabulary')renderVocabulary();
if(sheet==='story-tools')setText('metrics',generated.length+' generated tokens · about '+decode(generated.map(t=>t.id)).trim().split(/\s+/).filter(Boolean).length+' words');
if(sheet==='weight-tools')setText('edited-tag',Object.keys(overrides).length?Object.keys(overrides).length+' output connection(s) edited.':'All output connections use their trained weights.');
}
function renderExplorer(){
setText('explorer-connection',connectionName());setText('explorer-math',connectionMath()+' added to the score.');
const query=$('neuron-search').value.trim();$('memory-grid').replaceChildren();
for(let h=0;h<H;h++){
if(query&&!String(h+1).includes(query))continue;
const b=node('button',latest.hidden[h]<0?'negative':'');b.append(node('span','','#'+(h+1)),node('b','',signed(latest.hidden[h])));
b.setAttribute('aria-label',features[h]+', activation '+signed(latest.hidden[h]));b.setAttribute('aria-pressed',String(h===selectedH));
b.addEventListener('click',()=>{selectedH=h;renderExplorerSelection();scheduleSave();});b.dataset.neuron=h;$('memory-grid').append(b);
}
renderOutputSearch();
}
function renderExplorerSelection(){
setText('explorer-connection',connectionName());setText('explorer-math',connectionMath()+' added to the score.');
document.querySelectorAll('[data-neuron]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.neuron)===selectedH)));
}
function renderOutputSearch(){
const q=$('output-search').value.trim().toLowerCase();
const candidates=q?vocabulary.map((_,id)=>id).filter(id=>pieceLabel(id).toLowerCase().includes(q)).slice(0,24):ranked().slice(0,6).map(t=>t.id);
$('output-results').replaceChildren();
if(!candidates.length)$('output-results').append(node('p','note','No matching tokens. Try a shorter piece.'));
for(const id of candidates){const b=node('button','',pieceLabel(id)+' · '+probabilityText(latest.probabilities[id]));b.setAttribute('aria-pressed',String(selectedV===id));b.dataset.output=id;b.addEventListener('click',()=>{selectedV=id;renderExplorerSelection();$('output-results').querySelectorAll('button').forEach(el=>el.setAttribute('aria-pressed',String(Number(el.dataset.output)===id)));scheduleSave();});$('output-results').append(b);}
}
function renderVocabulary(){
const q=$('vocab-search').value.trim().toLowerCase(),matches=vocabulary.map((_,id)=>id).filter(id=>pieceLabel(id).toLowerCase().includes(q));
setText('vocab-count',matches.length+' matches'+(matches.length>80?' · showing the first 80. Type to narrow your search.':'.'));
fillTokens($('vocabulary'),matches.slice(0,80));
}
function showSheet(key,{push=true}={}){
const section=$('sheet-'+key);if(!section)return;
if(!dialog.open)sheetTrigger=document.activeElement;
if(running){stop('Paused while you explore');renderNow();}
sheet=key;document.querySelectorAll('.sheet-content').forEach(s=>s.hidden=s!==section);
setText('sheet-title',section.dataset.title);dialog.dataset.wide=String(['network','vocabulary'].includes(key));
if(!dialog.open)dialog.showModal();$('sheet-body').scrollTop=0;renderSheet();$('sheet-title').focus({preventScroll:true});
if(push)writeHistory('push',{slide:currentId(),sheet:key});
}
function dismissSheet({restoreFocus=true}={}){
sheet='';if(dialog.open)dialog.close();
if(restoreFocus&&sheetTrigger?.isConnected&&!sheetTrigger.closest('[inert]'))sheetTrigger.focus({preventScroll:true});
}
function closeSheet(){
if(history.state?.sheet){closingHistory=true;history.back();}else dismissSheet();
}
let afterClose=null;
function leaveSheet(action){afterClose=action;closeSheet();if(!closingHistory){const fn=afterClose;afterClose=null;fn?.();}}
window.addEventListener('popstate',event=>{
dismissSheet({restoreFocus:!afterClose});
goToPage(event.state?.slide||location.hash.slice(1)||'intro',{historyMode:'none',focus:!closingHistory});
if(event.state?.sheet)showSheet(event.state.sheet,{push:false});
closingHistory=false;if(afterClose){const action=afterClose;afterClose=null;action();}
});
dialog.addEventListener('cancel',e=>{e.preventDefault();closeSheet();});
dialog.addEventListener('click',e=>{const r=dialog.getBoundingClientRect();if(e.target===dialog&&(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom))closeSheet();});
$('close-sheet').addEventListener('click',closeSheet);
document.querySelectorAll('[data-open]').forEach(b=>{b.setAttribute('aria-haspopup','dialog');b.addEventListener('click',()=>showSheet(b.dataset.open));});
document.querySelectorAll('[data-switch]').forEach(b=>b.addEventListener('click',()=>{showSheet(b.dataset.switch,{push:false});writeHistory('replace',{slide:currentId(),sheet:b.dataset.switch});}));
document.querySelectorAll('a[href^="#"],[data-go]').forEach(link=>link.addEventListener('click',e=>{e.preventDefault();goToPage(link.dataset.go||link.hash.slice(1));}));
$('previous-page').addEventListener('click',()=>goToPage(activePage-1));
$('next-page').addEventListener('click',()=>goToPage(activePage===pageList.length-1?'playground':activePage+1));
pageList.forEach((p,i)=>{const b=node('button');b.append(node('small','',String(i+1).padStart(2,'0')),node('span','',p.dataset.title));b.addEventListener('click',()=>leaveSheet(()=>goToPage(i)));$('chapter-list').append(b);});
document.addEventListener('keydown',e=>{
if(dialog.open||e.defaultPrevented||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,textarea,select,button,a,summary,[contenteditable="true"]'))return;
if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();goToPage(activePage+1);}
if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();goToPage(activePage-1);}
});
// Vertical gestures always read. Optional horizontal swipes ignore interactive areas.
$('pages').addEventListener('touchstart',e=>{const t=e.touches[0];swipe=e.touches.length===1&&!e.target.closest('input,textarea,select,button,a,summary,details,#samples')?{x:t.clientX,y:t.clientY,time:performance.now()}:null;},{passive:true});
$('pages').addEventListener('touchend',e=>{if(!swipe||dialog.open)return;const t=e.changedTouches[0],dx=swipe.x-t.clientX,dy=swipe.y-t.clientY,elapsed=performance.now()-swipe.time;swipe=null;if(elapsed<700&&Math.abs(dx)>90&&Math.abs(dx)>Math.abs(dy)*2)goToPage(activePage+Math.sign(dx));},{passive:true});
$('pages').addEventListener('touchcancel',()=>{swipe=null;},{passive:true});
$('generate').addEventListener('click',generateStory);$('continue-generation').addEventListener('click',generateStory);$('step').addEventListener('click',()=>appendToken());
$('prompt').addEventListener('input',reset);$('seed').addEventListener('change',()=>{if(!$('seed').checkValidity()){$('seed').value='42';}reset();toast('New seed. Ready for a new story.');});
$('temperature').addEventListener('input',()=>{setText('temperature-value',Number($('temperature').value).toFixed(2));stop('Randomness updated');infer();render();});
$('limit').addEventListener('change',()=>{stop('Batch length updated');render();});
for(const id of ['slow','one-sentence'])$(id).addEventListener('change',scheduleSave);
document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>{const action=()=>{$('prompt').value=b.dataset.prompt;reset();goToPage('playground');};if(dialog.open)leaveSheet(action);else action();}));
$('reset').addEventListener('click',()=>leaveSheet(()=>{reset();goToPage('playground');}));
$('output-weight').addEventListener('input',e=>changeWeight(Number(e.target.value)));
$('weight-down').addEventListener('click',()=>changeWeight(weightAt(selectedV,selectedH)-.1));$('weight-up').addEventListener('click',()=>changeWeight(weightAt(selectedV,selectedH)+.1));
$('restore-connection').addEventListener('click',()=>{changeWeight(originalWeights[selectedV*H+selectedH]);toast('Connection restored to its trained weight.');});
$('restore').addEventListener('click',()=>{stop('Trained weights restored');model.output.set(originalWeights);overrides={};infer();render();toast('All trained weights restored.');scheduleSave();});
const editSelected=()=>leaveSheet(()=>goToPage('weights',{keepConnection:true}));
$('edit-connection').addEventListener('click',editSelected);$('explorer-edit').addEventListener('click',editSelected);
$('neuron-search').addEventListener('input',renderExplorer);$('output-search').addEventListener('input',renderOutputSearch);$('vocab-search').addEventListener('input',renderVocabulary);
function toast(message){setText('toast',message);clearTimeout(toastTimer);toastTimer=setTimeout(()=>setText('toast',''),3500);}
function downloadFile(content,name,type){const url=URL.createObjectURL(new Blob([content],{type})),a=node('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download').addEventListener('click',()=>{const html=ORIGINAL_HTML.replace(/(<script id="model-overrides" type="application\/json">)[\s\S]*?(<\/script>)/,(_,open,close)=>open+JSON.stringify({version:4,weights:overrides})+close);downloadFile(html,'pocket-lm.html','text/html;charset=utf-8');toast('Offline app saved with your weight edits.');});
$('download-model').addEventListener('click',()=>{downloadFile(JSON.stringify({...bundle,edits:{version:4,weights:overrides}}),'pocket-lm-model.json','application/json');toast('Model saved with your weight edits.');});
const storyText=()=>$('prompt').value+decode(allIds()).slice(decode(promptTokens.map(t=>t.id)).length);
$('save-text').addEventListener('click',()=>{downloadFile(storyText()+'\n','a-very-strange-christmas.txt','text/plain;charset=utf-8');toast('Story saved.');});
$('copy-story').addEventListener('click',async()=>{
const text=storyText();let copied=false;
try{await navigator.clipboard.writeText(text);copied=true;}
catch{
const field=node('textarea');field.value=text;field.readOnly=true;field.style.cssText='position:fixed;opacity:0;pointer-events:none';dialog.append(field);
try{field.focus();field.select();field.setSelectionRange(0,text.length);copied=document.execCommand('copy');}catch{}
field.remove();$('copy-story').focus({preventScroll:true});
}
toast(copied?'Story copied to clipboard.':'Copy unavailable. Use Save story as text.');
});
function scheduleSave(){if(!ready)return;clearTimeout(sessionTimer);sessionTimer=setTimeout(saveSession,250);}
function saveSession(){
if(!ready)return;
try{sessionStorage.setItem('pocket-lm-session-v1',JSON.stringify({slide:currentId(),prompt:$('prompt').value,seed:Number($('seed').value),temperature:Number($('temperature').value),limit:$('limit').value,slow:$('slow').checked,sentence:$('one-sentence').checked,generated,hidden:Array.from(memory.hidden),recent:memory.recent,edits:overrides,h:selectedH,v:selectedV}));}catch{/* Private browsing and file URLs may deny storage. */}
}
function restoreSession(){
try{
const saved=JSON.parse(sessionStorage.getItem('pocket-lm-session-v1'));if(!saved||typeof saved.prompt!=='string'||saved.prompt.length>2000)return null;
const validToken=id=>Number.isInteger(id)&&id>=0&&id<V;
if(!Number.isInteger(saved.seed)||saved.seed<0||saved.seed>4294967295||!Number.isFinite(saved.temperature)||saved.temperature<0||saved.temperature>1.5)return null;
if(!Array.isArray(saved.generated)||saved.generated.length>MAX_DOCUMENT||!saved.generated.every(t=>validToken(t.id)&&typeof t.forced==='boolean'))return null;
if(!Array.isArray(saved.hidden)||saved.hidden.length!==H||!saved.hidden.every(v=>Number.isFinite(v)&&Math.abs(v)<=1)||!Array.isArray(saved.recent)||saved.recent.length>8||!saved.recent.every(validToken))return null;
$('prompt').value=saved.prompt;$('seed').value=saved.seed;$('temperature').value=saved.temperature;
if(['32','64','128','256','512','1024'].includes(saved.limit))$('limit').value=saved.limit;
$('slow').checked=!!saved.slow;$('one-sentence').checked=!!saved.sentence;
applyOverrides({version:4,weights:saved.edits});promptTokens=tokenize(saved.prompt);generated=saved.generated;
memory={hidden:new Float32Array(saved.hidden),recent:saved.recent};rng=randomGenerator(saved.seed);for(const t of generated)if(!t.forced)rng();
infer();selectedH=Number.isInteger(saved.h)&&saved.h>=0&&saved.h<H?saved.h:0;selectedV=validToken(saved.v)?saved.v:0;
status=generated.length?'Your story is restored · continue whenever you like':'Ready when you are';return saved.slide;
}catch{return null;}
}
window.addEventListener('pagehide',saveSession);
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(running){stop('Paused while you were away');render();}saveSession();}});
setText('param-stat',number(parameterCount));setText('vocab-stat',number(V));setText('model-size',(bundle.stats.packedWeightBytes/1e6).toFixed(3)+' MB');setText('page-size',number(new TextEncoder().encode(ORIGINAL_HTML).length)+' bytes');
const training=bundle.training,holiday=training.holiday;
setText('training-summary',number(training.trainStories)+' base stories and '+(training.trainingTokens/1e6).toFixed(2)+' million tokens; then 46 descriptive passages, '+holiday.corpus.train.paragraphs+' Christmas stories, and 20 longer Christmas stories.');
setText('val-baseline',holiday.initialValidationLoss.toFixed(3));setText('val-loss',holiday.quantizedValidationLoss.toFixed(3));
const points=[{step:0,loss:holiday.initialValidationLoss},...holiday.history.map(e=>({step:e.adaptationStep,loss:e.christmasValidationLoss}))],maxStep=Math.max(...points.map(p=>p.step)),maxLoss=Math.max(...points.map(p=>p.loss));
function svg(tag,attrs){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n;}
for(const y of [20,65,110,150])$('loss-chart').append(svg('path',{d:'M 5 '+y+' H 495',stroke:'#e4e6df',fill:'none'}));
$('loss-chart').append(svg('path',{d:points.map((p,i)=>(i?'L':'M')+' '+(5+p.step/maxStep*490).toFixed(1)+' '+(150-p.loss/maxLoss*140).toFixed(1)).join(' '),fill:'none',stroke:'#a83e27','stroke-width':3}));setText('chart-end',number(maxStep)+' training steps');
setText('architecture','Vocabulary: '+number(V)+' BPE pieces\nEmbedding: '+D+' values per token\nMemory: '+H+' GRU activations\nParameters: '+number(parameterCount)+'\nPacked weights: '+number(bundle.stats.packedWeightBytes)+' bytes');
reset();const savedSlide=restoreSession();setText('temperature-value',Number($('temperature').value).toFixed(2));ready=true;
const requested=location.hash.slice(1),initial=pageList.some(p=>p.id===requested)?requested:pageList.some(p=>p.id===savedSlide)?savedSlide:'intro';
goToPage(initial,{historyMode:'replace',focus:false,keepConnection:true});
window.PocketLM={bundle,model,modelStep,modelLogits,softmax,loadWeights,forward,predict,promptMemory,initialMemory,advance,tokenize,decode,randomGenerator,sample,weightAt,setWeight,applyOverrides};

const sampleStories=JSON.parse($('example-data').textContent);let sampleIndex=0,sampleTouch=null;
function renderSample(){const [title,text]=sampleStories[sampleIndex];setText('sample-title',title);setText('sample-text',text);const dots=$('sample-count');dots.setAttribute('aria-label','Story '+(sampleIndex+1)+' of '+sampleStories.length);dots.innerHTML=sampleStories.map((_,i)=>'<span aria-hidden="true" class="'+(i===sampleIndex?'selected':'')+'"></span>').join('');}
function moveSample(delta){sampleIndex=(sampleIndex+delta+sampleStories.length)%sampleStories.length;renderSample();$('samples').scrollIntoView({block:'start'});}
$('sample-prev').onclick=()=>moveSample(-1);$('sample-next').onclick=()=>moveSample(1);
$('samples').addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();moveSample(e.key==='ArrowRight'?1:-1);}});
$('samples').addEventListener('touchstart',e=>{sampleTouch=e.touches.length===1?e.touches[0]:null;},{passive:true});
$('samples').addEventListener('touchend',e=>{if(!sampleTouch)return;const t=e.changedTouches[0],dx=sampleTouch.clientX-t.clientX,dy=sampleTouch.clientY-t.clientY;sampleTouch=null;if(Math.abs(dx)>50&&Math.abs(dx)>Math.abs(dy)*2)moveSample(Math.sign(dx));},{passive:true});
$('samples').addEventListener('touchcancel',()=>sampleTouch=null,{passive:true});renderSample();
