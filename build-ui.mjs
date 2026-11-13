// Rebuild the standalone UI without changing any trained data or numerical code.
import {readFileSync,writeFileSync} from 'node:fs';
import {minify} from 'terser';
import CleanCSS from 'clean-css';
const base=new URL('./',import.meta.url),read=p=>readFileSync(new URL(p,base),'utf8');
const old=read('index.html');
const checkpoint=old.match(/<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const kernel=old.match(/<script>\s*([\s\S]*?)<\/script>/)[1].split('// MODEL END:')[0];
const css=new CleanCSS({level:2}).minify(read('ui/style.css')).styles;
const body=read('ui/body.html').replace(/>\s+</g,'><').trim();
const {code:app}=await minify(read('ui/app.js'),{ecma:2022,toplevel:true,compress:true,mangle:{reserved:['H','D','V','model','overrides','bundle','vocabulary','features','tokenIds','originalWeights','mergeRanks','tokenCache','UNK','PARAGRAPH','parameterCount','ORIGINAL_HTML']},format:{comments:false}});
let compiledCSS=css,compiledBody=body,compiledApp=app;
// Shorten only a fixed set of UI tokens. Never transform checkpoint or kernel.
const names=['page-content','page-footer','page-head','sheet-content','flow-card','flow-label','prob-list','prob-top','prob-track','prob-fill','token-pill','lesson-step','lesson-steps','chapter-list','memory-grid','activation-value','top-actions','welcome-actions','welcome-art','nav-footer','primary','text-button','featured','takeaway','comparison','sheet-header','sheet-body','search-results','vocab-results','prompt-content','range-row','weight-number','chapters-button','check-row','setting-row','brandmark','predictions-more','training-results','training-data','about-model'];
names.forEach((name,i)=>{const pattern=new RegExp('\\b'+name+'\\b','g'),alias='u'+i;compiledCSS=compiledCSS.replace(pattern,alias);compiledBody=compiledBody.replace(pattern,alias);compiledApp=compiledApp.replace(pattern,alias);});
const html='<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, interactive-widget=resizes-content"><meta name="theme-color" content="#f6f3ee"><meta name="description" content="Meet Pocket LM. Make a story, follow a prediction, and explore a real trained language model running entirely on your device."><title>Pocket LM — A little model, a lot to discover</title><link rel="icon" href="data:,"><style>'+compiledCSS+'</style>\n</head>\n<body>\n'+compiledBody+'\n<script id="model-overrides" type="application/json">{}</script>\n<script id="model-data" type="application/json">'+checkpoint+'</script>\n<script>\n'+kernel+'// MODEL END: tests run the exact shipped kernel and checkpoint.\n'+compiledApp+'\n</script>\n</body>\n</html>\n';
if(Buffer.byteLength(html)>=1_000_000)throw Error('UI exceeds the standalone 1 MB budget: '+Buffer.byteLength(html));
writeFileSync(new URL('index.html',base),html);
console.log('Built standalone page: '+Buffer.byteLength(html)+' bytes. Checkpoint and kernel preserved.');
