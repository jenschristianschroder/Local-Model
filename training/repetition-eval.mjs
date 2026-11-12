// Offline evaluation of the exact browser kernel; never imported by the page.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
const [htmlPath,outputPath,suite='regression']=process.argv.slice(2);
const html=readFileSync(htmlPath,'utf8');
const encoded=html.match(/<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const bundle=JSON.parse(encoded),source=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1].split('// MODEL END:')[0];
const ctx=vm.createContext({atob,document:{head:{outerHTML:''},body:{outerHTML:''},getElementById:id=>({textContent:id==='model-data'?encoded:'{}'})}});
vm.runInContext(source+';globalThis.m={promptMemory,tokenize,predict,sample,randomGenerator,advance,decode};',ctx);
const m=ctx.m,results=[];
const prompts=suite==='additional'?['Santa stared at a singing potato','Christmas dinner began with','The bells in the workshop','In the snowy village']:['Santa opened the fridge','The Christmas pudding','A reindeer','On Christmas Eve','The elf found a strange letter','Mrs Claus looked inside the box'];
for(const temperature of [0,.55,.8])for(const prompt of prompts)for(const seed of temperature===0?[42]:[42,137]){
  const prefix=m.tokenize(prompt).map(x=>x.id),memory=m.promptMemory(prefix),random=m.randomGenerator(seed),ids=[];
  for(let i=0;i<1024;i++){const id=m.sample(m.predict(memory,{temperature}).probabilities,random);ids.push(id);m.advance(memory,id);}
  const text=m.decode([...prefix,...ids]),words=m.decode(ids).split(/\s+/).filter(Boolean);
  const repeats=n=>{const seen=new Set();let repeated=0;for(let i=0;i<=words.length-n;i++){const gram=words.slice(i,i+n).join(' ');if(seen.has(gram))repeated++;seen.add(gram);}return repeated/Math.max(1,words.length-n+1);};
  const paragraphs=text.split(/\n\s*\n/).map(x=>x.trim()).filter(x=>x.split(/\s+/).length>=20);
  const duplicateParagraphs=paragraphs.length-new Set(paragraphs).size;
  results.push({prompt,temperature,seed,tokens:ids.length,words:words.length,repeated8:repeats(8),repeated32:repeats(32),duplicateParagraphs,text});
}
const summary=[0,.55,.8].map(temperature=>{const rows=results.filter(x=>x.temperature===temperature);return{temperature,cases:rows.length,meanRepeated8:rows.reduce((n,x)=>n+x.repeated8,0)/rows.length,meanRepeated32:rows.reduce((n,x)=>n+x.repeated32,0)/rows.length,casesWithDuplicateParagraphs:rows.filter(x=>x.duplicateParagraphs>0).length};});
const report={checkpointSha256:createHash('sha256').update(encoded).digest('hex'),suite,tokensPerCase:1024,description:'Fraction of generated word n-grams already seen earlier in the same continuation. Exact repeated paragraphs count only paragraphs of at least 20 words. This is a narrow regression suite, not a measure of overall story quality.',summary,results};
if(outputPath)writeFileSync(outputPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({checkpointSha256:report.checkpointSha256,summary},null,2));
