// Offline weight authoring. This file is not loaded by the browser.
// All language-specific decisions below compile to finite numeric parameters.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const POS = ['det','det_s','det_p','adj','ns','np','mass','ps','pp','ts','tp','vs','vp','bs','bp','modal','prep','adv','join','intro','punct'];
const TOPICS = ['technology','nature','night sky','adventure','everyday life','ideas'];
const SEM = ['actor','machine','place','water','sky','idea','thing','plant'];
const BASE_D = POS.length + TOPICS.length + SEM.length * 2;
const vocabulary=[], embeddings=[], tokenIds=new Map(), metadata=[];
function add(words, parts, topic=null, semantic=null, relation=null) {
  for(const word of words.split(' ')) {
    let id=tokenIds.get(word);
    if(id===undefined){id=vocabulary.length;tokenIds.set(word,id);vocabulary.push(word);embeddings.push(new Float64Array(BASE_D));metadata.push({});}
    const e=embeddings[id];
    for(const part of parts)e[POS.indexOf(part)]=1;
    if(topic!==null)e[POS.length+TOPICS.indexOf(topic)]=1;
    if(semantic!==null)e[POS.length+TOPICS.length+SEM.indexOf(semantic)]=1;
    if(relation)for(const [name,value] of Object.entries(relation))e[POS.length+TOPICS.length+SEM.length+SEM.indexOf(name)]=value;
  }
}
add('<bos> <unk>',[]);
add('the its our their', ['det']); add('a an this that every another', ['det_s']); add('these those some many', ['det_p']);
add('little small large old new quiet gentle curious bright dark strange distant warm cool hidden patient simple silent lovely soft careful brave tiny beautiful mysterious', ['adj']);
add('silver electric mechanical clever precise helpful copper golden', ['adj'],'technology');
add('green wild leafy fresh ancient wooden peaceful deep misty rainy sunlit blooming', ['adj'],'nature');
add('luminous endless pale starlit cosmic blue violet shining', ['adj'],'night sky');
add('forgotten winding narrow dusty secret empty unfamiliar rocky', ['adj'],'adventure');
add('familiar comfortable cozy friendly cheerful open colorful', ['adj'],'everyday life');
add('wonderful unusual possible surprising thoughtful hopeful creative', ['adj'],'ideas');
function nouns(pairs,topic,semantic){for(const pair of pairs.split(' ')){const [singular,plural]=pair.split('/');add(singular,['ns'],topic,semantic);if(plural)add(plural,['np'],topic,semantic);}}
nouns('robot/robots machine/machines computer/computers engine/engines clock/clocks sensor/sensors drone/drones telescope/telescopes','technology','machine');
nouns('circuit/circuits device/devices signal/signals screen/screens wire/wires tool/tools wheel/wheels lantern/lanterns bridge/bridges','technology','thing');
nouns('cat/cats bird/birds fox/foxes owl/owls deer butterfly/butterflies rabbit/rabbits bee/bees','nature','actor');
nouns('forest/forests garden/gardens meadow/meadows valley/valleys hill/hills mountain/mountains field/fields island/islands','nature','place');
nouns('river/rivers stream/streams lake/lakes ocean/oceans pool/pools','nature','water');
nouns('flower/flowers tree/trees leaf/leaves seed/seeds branch/branches root/roots','nature','plant');
nouns('stone/stones','nature','thing');
nouns('moon/moons star/stars planet/planets comet/comets cloud/clouds constellation/constellations galaxy/galaxies sun/suns','night sky','sky');
nouns('horizon/horizons universe/universes','night sky','place');
nouns('traveler/travelers explorer/explorers captain/captains sailor/sailors','adventure','actor');
nouns('path/paths road/roads trail/trails village/villages tower/towers castle/castles cave/caves harbor/harbors','adventure','place');
nouns('map/maps compass/compasses boat/boats ship/ships key/keys gate/gates treasure/treasures','adventure','thing');
nouns('friend/friends child/children artist/artists musician/musicians neighbor/neighbors teacher/teachers','everyday life','actor');
nouns('house/houses room/rooms kitchen/kitchens library/libraries workshop/workshops courtyard/courtyards','everyday life','place');
nouns('window/windows door/doors table/tables cup/cups book/books letter/letters basket/baskets painting/paintings song/songs','everyday life','thing');
nouns('dream/dreams idea/ideas story/stories question/questions answer/answers memory/memories pattern/patterns possibility/possibilities secret/secrets thought/thoughts promise/promises discovery/discoveries','ideas','idea');
add('water rain dew',['mass'],'nature','water');add('grass moss earth',['mass'],'nature','thing');
add('moonlight sunlight starlight darkness',['mass'],'night sky','sky');add('silence music warmth',['mass'],'everyday life','idea');
add('hope wonder time knowledge peace',['mass'],'ideas','idea');add('light energy',['mass'],'technology','thing');
add('it she he someone',['ps']);add('we they you',['pp']);
add('she he someone',['ps'],null,'actor');add('we you',['pp'],null,'actor');
const perceive={actor:.5,machine:.3,place:.4,water:.4,sky:.5,idea:.3,thing:.5};
const make={actor:-1,machine:1.4,place:.4,water:-1,sky:-1,idea:.4,thing:1.3};
const think={actor:.1,machine:-.4,place:0,water:-.4,sky:.1,idea:1.7,thing:.2};
function verbs(pairs,parts,topic,relation,subjectWeights){for(const pair of pairs.split(' ')){const forms=pair.split('/');forms.forEach((word,i)=>{add(word,[parts[i]],topic,null,relation);metadata[tokenIds.get(word)].subjectWeights=subjectWeights;});}}
const active={actor:1.8,machine:1.2,place:-6,water:-6,sky:-6,idea:-4,thing:-6,plant:-6};
const maker={actor:1.7,machine:2.4,place:-6,water:-6,sky:-6,idea:-5,thing:-6,plant:-6};
const reflective={actor:1.4,machine:.4,place:-6,water:-6,sky:-5,idea:-3,thing:-6,plant:-6};
const broad={actor:.1,machine:.2,place:1.6,water:1.5,sky:1.1,idea:1,thing:1,plant:.8};
verbs('finds/find sees/see notices/notice watches/watch follows/follow explores/explore discovers/discover',['ts','tp'],null,perceive,active);
verbs('builds/build repairs/repair designs/design tests/test creates/create',['ts','tp'],'technology',make,maker);
verbs('imagines/imagine remembers/remember understands/understand loves/love considers/consider',['ts','tp'],'ideas',think,reflective);
verbs('holds/hold carries/carry hides/hide reveals/reveal reflects/reflect welcomes/welcome',['ts','tp'],null,perceive,broad);
verbs('wanders/wander walks/walk waits/wait rests/rest listens/listen returns/return smiles/smile',['vs','vp'],null,null,active);
verbs('sleeps/sleep wakes/wake plays/play',['vs','vp'],'nature',null,active);
verbs('hums/hum clicks/click works/work moves/move',['vs','vp'],'technology',null,maker);
verbs('glows/glow shines/shine sparkles/sparkle',['vs','vp'],'night sky',null,{actor:-1,machine:.5,place:.8,water:1.4,sky:2,idea:.3,thing:1});
verbs('flows/flow ripples/ripple',['vs','vp'],'nature',null,{actor:-6,machine:-6,place:-6,water:4,sky:-6,idea:-6,thing:-6,plant:-6});
verbs('grows/grow blooms/bloom',['vs','vp'],'nature',null,{actor:-6,machine:-6,place:1.2,water:-6,sky:-6,idea:-1,thing:-6,plant:3});
verbs('drifts/drift',['vs','vp'],'night sky',null,{actor:-1,machine:-1,place:-3,water:1,sky:2.5,idea:1,thing:.2});
verbs('dreams/dream',['vs','vp'],'ideas',null,reflective);
add('is was',['bs']);add('are were',['bp']);add('can could might will',['modal']);
add('in near beside under above beyond through across around within toward outside',['prep']);
add('quietly slowly softly gently patiently calmly brightly peacefully again together',['adv']);
add('and but while because',['join']);add('later meanwhile nearby sometimes soon today',['intro']);add('. !',['punct']);
const V=vocabulary.length, BOS=tokenIds.get('<bos>'), UNK=tokenIds.get('<unk>');
const has=(id,part)=>embeddings[id][POS.indexOf(part)]>0;

const stateNames=['Sentence start','Subject after the','Singular subject phrase','Plural subject phrase','Singular subject','Plural subject','Object start','Object after the','Singular object phrase','Plural object phrase','Object complete','Intransitive verb complete','After is / are','Predicate adjective','Place phrase start','Place after the','Singular place phrase','Plural place phrase','Place phrase complete','Adverb complete','After a modal','New sentence','After an introduction'];
const S=stateNames.length;
const ST={start:0,sub:1,subS:2,subP:3,singular:4,plural:5,obj:6,objDet:7,objS:8,objP:9,objDone:10,verbDone:11,copula:12,pred:13,prep:14,prepDet:15,prepS:16,prepP:17,prepDone:18,adv:19,modal:20,boundary:21,intro:22};
const ADVERBS='quietly slowly softly gently patiently calmly brightly peacefully again together'.split(' ');
const manners=new Map();
function mannerWeights(words,adverbs){for(const word of words.split(' '))manners.set(word,new Set(adverbs.split(' ')));}
mannerWeights('hums hum clicks click','quietly softly gently again together');
mannerWeights('wanders wander walks walk returns return moves move drifts drift','slowly quietly gently peacefully again together');
mannerWeights('waits wait listens listen','quietly patiently calmly again together');
mannerWeights('rests rest sleeps sleep','quietly peacefully calmly again together');
mannerWeights('wakes wake','slowly quietly gently again together');
mannerWeights('plays play','quietly gently calmly again together');
mannerWeights('smiles smile','softly gently calmly brightly again together');
mannerWeights('glows glow shines shine sparkles sparkle','brightly softly gently again together');
mannerWeights('grows grow blooms bloom','slowly again together');
mannerWeights('flows flow ripples ripple','slowly gently quietly calmly again together');
mannerWeights('works work','quietly slowly patiently calmly again together');
mannerWeights('dreams dream','peacefully again together');

// Embedding entries are authored here, then frozen into sparse numeric rows.
// The browser never receives these POS tags or lexical compatibility tables.
const recoverOffset=BASE_D,mannerOffset=recoverOffset+S,unknownOffset=mannerOffset+ADVERBS.length;
const articleOffset=unknownOffset+1,identityOffset=articleOffset+2,D=identityOffset+V;
const E=embeddings.map((e,v)=>{
  const row=new Float64Array(D);row.set(e);
  const recovery=has(v,'punct')?ST.boundary:has(v,'np')||has(v,'pp')?ST.plural:has(v,'ns')||has(v,'mass')||has(v,'ps')?ST.singular:has(v,'ts')||has(v,'tp')?ST.obj:has(v,'vs')||has(v,'vp')?ST.verbDone:has(v,'bs')||has(v,'bp')?ST.copula:has(v,'prep')?ST.prep:has(v,'adj')?ST.sub:ST.start;
  if(v!==UNK)row[recoverOffset+recovery]=1;
  row[unknownOffset]=v===UNK?1:0;
  row[articleOffset]=vocabulary[v]==='a'?1:0;row[articleOffset+1]=vocabulary[v]==='an'?1:0;
  ADVERBS.forEach((word,j)=>{row[mannerOffset+j]=manners.get(vocabulary[v])?.has(word)?1:0;});
  // Shift signed object preferences into positive ReLU memory channels.
  if(has(v,'ts')||has(v,'tp'))for(let j=0;j<SEM.length;j++)row[POS.length+TOPICS.length+SEM.length+j]+=1;
  row[identityOffset+v]=1;
  return row;
});

const features=[...stateNames],F={};
function channels(key,names){F[key]=features.length;features.push(...names);return F[key];}
channels('theme',TOPICS.map(x=>'Theme: '+x));
channels('slowTheme',TOPICS.map(x=>'Long theme: '+x));
channels('subject',SEM.map(x=>'Subject memory: '+x));
channels('object',SEM.map(x=>'Object memory: '+x));
channels('singular',['Singular subject memory']);channels('plural',['Plural subject memory']);
channels('alone',['Single / unknown subject']);
channels('manner',ADVERBS.map(x=>'Verb affinity: '+x));
channels('article',['Last input: a','Last input: an']);
channels('adjectives',['Consecutive adjectives']);channels('length',['Sentence length memory']);
channels('trace',vocabulary.map(x=>'Recent activation: '+x));
channels('subjectScore',SEM.map(x=>'Subject: '+x));channels('objectScore',SEM.map(x=>'Object: '+x));
channels('mismatch',ADVERBS.map(x=>'Verb mismatch: '+x));
channels('adjectivePressure',['Extra adjectives']);channels('lengthPressure',['Long sentence']);
const H=features.length,base=H+D,cell=[],cellBias=[],readout=new Array(H);
// A sparse, feed-forward ReLU cell with recurrent inputs. Every connection is
// an index and a finite weight. The runtime evaluates this graph generically.
function neuron(bias,edges=[]){const row=[];for(const[index,weight]of edges)if(weight!==0)row.push(index,weight);const n=base+cell.length;cell.push(row);cellBias.push(bias);return n;}
const input=j=>H+j;
const pos=p=>input(POS.indexOf(p));
const sum=indices=>neuron(0,indices.map(index=>[index,1]));
const cap=(n,maximum)=>neuron(0,[[n,1],[neuron(-maximum,[[n,1]]),-1]]);
const nextState=Array.from({length:S},()=>[]),transitionGates=[];
function transition(from,part,to){const n=neuron(-1,[...from.map(s=>[s,1]),[pos(part),1]]);nextState[to].push(n);transitionGates.push(n);}
const starts=[ST.start,ST.boundary,ST.intro];
transition(starts,'det',ST.sub);transition(starts,'det_s',ST.subS);transition(starts,'det_p',ST.subP);
transition([...starts,ST.sub,ST.subS],'ns',ST.singular);transition([...starts,ST.sub,ST.subP],'np',ST.plural);transition([...starts,ST.sub],'mass',ST.singular);
transition(starts,'ps',ST.singular);transition(starts,'pp',ST.plural);
for(const s of [ST.sub,ST.subS,ST.subP])transition([s],'adj',s);
transition([ST.start,ST.boundary],'intro',ST.intro);
transition([ST.singular],'ts',ST.obj);transition([ST.plural,ST.modal],'tp',ST.obj);
transition([ST.singular],'vs',ST.verbDone);transition([ST.plural,ST.modal],'vp',ST.verbDone);
transition([ST.singular],'bs',ST.copula);transition([ST.plural],'bp',ST.copula);
transition([ST.singular,ST.plural],'modal',ST.modal);
transition([ST.obj,ST.copula],'det',ST.objDet);transition([ST.obj,ST.copula],'det_s',ST.objS);transition([ST.obj,ST.copula],'det_p',ST.objP);
transition([ST.obj,ST.objDet,ST.objS],'ns',ST.objDone);transition([ST.obj,ST.objDet,ST.objP],'np',ST.objDone);transition([ST.obj,ST.objDet],'mass',ST.objDone);
for(const s of [ST.objDet,ST.objS,ST.objP])transition([s],'adj',s);
transition([ST.copula],'adj',ST.pred);
const completes=[ST.objDone,ST.verbDone,ST.pred,ST.prepDone,ST.adv];
transition(completes,'prep',ST.prep);transition(completes,'join',ST.start);transition([ST.verbDone],'adv',ST.adv);
transition([ST.prep],'det',ST.prepDet);transition([ST.prep],'det_s',ST.prepS);transition([ST.prep],'det_p',ST.prepP);
transition([ST.prep,ST.prepDet,ST.prepS],'ns',ST.prepDone);transition([ST.prep,ST.prepDet,ST.prepP],'np',ST.prepDone);transition([ST.prep,ST.prepDet],'mass',ST.prepDone);
for(const s of [ST.prepDet,ST.prepS,ST.prepP])transition([s],'adj',s);
transition(Array.from({length:S},(_,i)=>i),'punct',ST.boundary);
for(let s=0;s<S;s++){const n=neuron(-1,[[s,1],[input(unknownOffset),1]]);nextState[s].push(n);transitionGates.push(n);}
const unmatched=neuron(1,transitionGates.map(n=>[n,-1]));
for(let s=0;s<S;s++){
  nextState[s].push(neuron(-1,[[unmatched,1],[input(recoverOffset+s),1]]));
}
// Competing ReLU activations resolve ambiguous embeddings such as "secret"
// (adjective/noun). Lateral inhibition keeps recurrent syntax activity bounded.
// These are numeric connections in the cell, not a runtime parser decision.
const stateActivity=nextState.map(nodes=>cap(sum(nodes),1));
for(let s=0;s<S;s++)readout[s]=neuron(0,[[stateActivity[s],1],...stateActivity.slice(0,s).map(n=>[n,-1])]);
const subjectNow=sum([readout[ST.singular],readout[ST.plural]]);
const newClause=sum([readout[ST.start],readout[ST.boundary]]);
const numberReset=sum([subjectNow,newClause]);
readout[F.singular]=sum([neuron(0,[[F.singular,1],[numberReset,-1]]),readout[ST.singular]]);
readout[F.plural]=sum([neuron(0,[[F.plural,1],[numberReset,-1]]),readout[ST.plural]]);
readout[F.alone]=neuron(1,[[readout[F.plural],-1]]);
const verbNow=sum([readout[ST.obj],readout[ST.verbDone],readout[ST.copula]]);
for(let j=0;j<ADVERBS.length;j++){
  readout[F.manner+j]=sum([neuron(0,[[F.manner+j,1],[verbNow,-1]]),neuron(-1,[[input(mannerOffset+j),1],[verbNow,1]])]);
  readout[F.mismatch+j]=neuron(1,[[readout[F.manner+j],-1]]);
}
const typedSubject=neuron(-1,[[subjectNow,1],...SEM.map((_,j)=>[input(POS.length+TOPICS.length+j),1])]);
const expectVerb=sum([readout[ST.singular],readout[ST.plural],readout[ST.modal]]);
const expectObject=sum([readout[ST.obj],readout[ST.objDet],readout[ST.objS],readout[ST.objP]]);
for(let j=0;j<SEM.length;j++){
  readout[F.subject+j]=sum([neuron(0,[[F.subject+j,1],[typedSubject,-1]]),neuron(-1,[[input(POS.length+TOPICS.length+j),1],[subjectNow,1]])]);
  readout[F.object+j]=sum([neuron(0,[[F.object+j,1],[readout[ST.obj],-3]]),neuron(-3,[[input(POS.length+TOPICS.length+SEM.length+j),1],[readout[ST.obj],3]])]);
  readout[F.subjectScore+j]=neuron(-1,[[readout[F.subject+j],1],[expectVerb,1]]);
  readout[F.objectScore+j]=neuron(-3,[[readout[F.object+j],1],[expectObject,3]]);
}
for(let j=0;j<TOPICS.length;j++){
  readout[F.theme+j]=cap(neuron(0,[[F.theme+j,.94],[input(POS.length+j),.45]]),1);
  readout[F.slowTheme+j]=cap(neuron(0,[[F.slowTheme+j,.995],[input(POS.length+j),.08]]),1);
}
for(let j=0;j<2;j++)readout[F.article+j]=neuron(0,[[input(articleOffset+j),1]]);
readout[F.adjectives]=cap(neuron(-3,[[F.adjectives,1],[pos('adj'),4]]),3);
readout[F.adjectivePressure]=neuron(-1,[[readout[F.adjectives],1]]);
readout[F.length]=neuron(0,[[cap(neuron(1,[[F.length,1]]),64),1],[pos('punct'),-64]]);
readout[F.lengthPressure]=neuron(-12,[[readout[F.length],1]]);
for(let v=0;v<V;v++)readout[F.trace+v]=neuron(0,[[F.trace+v,.94],[input(identityOffset+v),.35]]);

// The output layer produces FINITE scores for EVERY token. There is no mask.
// A preference of -24 is an ordinary editable weight, not a hard prohibition.
const W=Array.from({length:V},()=>new Float64Array(H)),bias=new Float64Array(V).fill(-30);
bias[BOS]=-80;bias[UNK]=-80;
function score(states,parts,value){for(let v=2;v<V;v++)if(parts.some(p=>has(v,p)))for(const s of states)W[v][s]=30+value;}
function wordScore(state,word,value){W[tokenIds.get(word)][state]=30+value;}
score(starts,['det'],8.5);score(starts,['det_s'],7.6);score(starts,['det_p'],6.5);score(starts,['ps','pp'],7.4);score(starts,['np','mass'],3.5);
score([ST.start,ST.boundary],['intro'],6.7);
for(const s of starts){wordScore(s,'the',10);wordScore(s,'a',8.4);wordScore(s,'an',7);wordScore(s,'it',8.4);wordScore(s,'they',7.6);}
score([ST.sub,ST.objDet,ST.prepDet],['ns','np','mass'],8);
score([ST.subS,ST.objS,ST.prepS],['ns'],8.1);score([ST.subP,ST.objP,ST.prepP],['np'],8.1);
score([ST.sub,ST.subS,ST.subP,ST.objDet,ST.objS,ST.objP,ST.prepDet,ST.prepS,ST.prepP],['adj'],6.3);
score([ST.singular],['ts','vs'],7.5);score([ST.plural],['tp','vp'],7.5);score([ST.modal],['tp','vp'],7.9);
score([ST.singular],['bs'],8.8);score([ST.plural],['bp'],8.8);score([ST.singular,ST.plural],['modal'],7.3);
score([ST.obj,ST.copula],['det','det_s'],8.6);score([ST.obj,ST.copula],['det_p'],6.4);
wordScore(ST.obj,'the',9.7);wordScore(ST.obj,'a',9.2);wordScore(ST.obj,'an',7.5);wordScore(ST.copula,'a',8.8);wordScore(ST.copula,'an',7.1);
score([ST.obj],['np','mass'],6.2);score([ST.copula],['adj'],8.2);
score(completes,['punct'],9.5);score(completes,['prep'],7);score(completes,['join'],7.1);
for(const s of completes){wordScore(s,'.',9.5);wordScore(s,'!',6);wordScore(s,'and',8);wordScore(s,'while',7.1);wordScore(s,'because',6.5);}
score([ST.verbDone],['adv'],8.5);score([ST.prepDone,ST.adv],['prep'],4.5);
score([ST.prep],['det','det_s'],8.3);score([ST.prep],['det_p'],6.1);score([ST.prep],['mass'],5.8);
wordScore(ST.prep,'the',10);wordScore(ST.prep,'a',8.7);wordScore(ST.prep,'an',6.9);
for(let v=2;v<V;v++){
  const e=embeddings[v];
  for(let j=0;j<TOPICS.length;j++){W[v][F.theme+j]=e[POS.length+j]*3;W[v][F.slowTheme+j]=e[POS.length+j]*1.4;}
  for(let j=0;j<SEM.length;j++){W[v][F.subjectScore+j]=metadata[v].subjectWeights?.[SEM[j]]||0;W[v][F.objectScore+j]=e[POS.length+TOPICS.length+j]*2.1;}
  if(has(v,'ns')||has(v,'np')||has(v,'mass'))for(const s of [ST.prepDet,ST.prepS,ST.prepP])if(W[v][s])W[v][s]+=e[POS.length+TOPICS.length+2]*1.5+e[POS.length+TOPICS.length+3]*.8;
  // Finite input/article and adjective preferences in the output tensor.
  W[v][F.article+(/^[aeiou]/.test(vocabulary[v])?0:1)]=-24;
  if(has(v,'adj'))W[v][F.adjectivePressure]=-16;
  if(has(v,'punct'))W[v][F.lengthPressure]=.38;
  if(has(v,'join')||has(v,'prep'))W[v][F.lengthPressure]=-.38;
  const content=['ns','np','mass','adj','ts','tp','vs','vp'].some(p=>has(v,p));
  W[v][F.trace+v]=content?-2.4:-.4;
}
ADVERBS.forEach((word,j)=>{W[tokenIds.get(word)][F.mismatch+j]=-24;});
W[tokenIds.get('together')][F.alone]=-24;
const initial=new Array(H).fill(0);initial[ST.start]=1;initial[F.alone]=1;
for(let j=0;j<ADVERBS.length;j++)initial[F.mismatch+j]=1;
function sparse(row){const result=[];for(let i=0;i<row.length;i++)if(row[i]!==0)result.push(i,row[i]);return result;}
export const checkpoint={format:3,width:H,embeddingWidth:D,embedding:E.map(sparse),cell,cellBias,readout,initial,output:W.map(sparse),outputBias:Array.from(bias)};
export const labels={vocabulary,features,stateNames,topics:TOPICS,semantics:SEM,adverbs:ADVERBS,offsets:F,states:ST,bos:BOS,unk:UNK};
const nonzero=checkpoint.embedding.reduce((n,r)=>n+r.length/2,0)+cell.reduce((n,r)=>n+r.length/2,0)+cellBias.filter(Boolean).length+checkpoint.output.reduce((n,r)=>n+r.length/2,0)+bias.filter(Boolean).length;
labels.stats={nonzero,neurons:cell.length,width:H,embeddingWidth:D,vocabulary:V};

// Keep a self-contained HTML artifact. This authoring script is reproducible,
// but the browser only loads the frozen checkpoint and a numeric-only kernel.
if(process.argv[1]===fileURLToPath(import.meta.url)){
  const path=new URL('./index.html',import.meta.url),html=readFileSync(path,'utf8');
  const payload=JSON.stringify({checkpoint,labels});
  const pattern=/(<script id="model-data" type="application\/json">)[\s\S]*?(<\/script>)/;
  if(!pattern.test(html))throw Error('Missing model-data block');
  const updated=html.replace(pattern,(_,open,close)=>open+payload+close);
  if(process.argv.includes('--check')){if(updated!==html)throw Error('The embedded weights are stale. Run node build-model.mjs.');}
  else writeFileSync(path,updated);
  console.log(JSON.stringify(labels.stats));
}

