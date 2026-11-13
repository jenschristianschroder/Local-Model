// GRU arithmetic and checkpoint decoder from Pocket LM's standalone kernel.
// Generic GRU arithmetic. No word names, syntax tags, masks, or repair rules.
function linear(matrix,bias,vector,rows,columns){
  const result=new Float32Array(rows);
  for(let row=0;row<rows;row++){let sum=bias?bias[row]:0;const offset=row*columns;for(let j=0;j<columns;j++)sum+=matrix[offset+j]*vector[j];result[row]=sum;}
  return result;
}
function modelStep(weights,previous,token){
  const {hidden:H,embedding:E}=weights.config;
  const embedding=weights.embedding.subarray(token*E,(token+1)*E);
  const input=linear(weights.input,weights.inputBias,embedding,3*H,E);
  const recurrent=linear(weights.recurrent,weights.recurrentBias,previous,3*H,H);
  const next=new Float32Array(H);
  for(let j=0;j<H;j++){
    const reset=1/(1+Math.exp(-(input[j]+recurrent[j])));
    const update=1/(1+Math.exp(-(input[H+j]+recurrent[H+j])));
    const candidate=Math.tanh(input[2*H+j]+reset*recurrent[2*H+j]);
    next[j]=(1-update)*candidate+update*previous[j];
  }
  return next;
}
function modelLogits(weights,hidden){return linear(weights.output,weights.outputBias,hidden,weights.config.vocabulary,weights.config.hidden);}
function softmax(logits,temperature=1){
  const p=new Float64Array(logits.length),maximum=Math.max(...logits);
  if(temperature===0){p[logits.indexOf(maximum)]=1;return p;}
  let total=0;for(let i=0;i<logits.length;i++){p[i]=Math.exp((logits[i]-maximum)/temperature);total+=p[i];}
  for(let i=0;i<p.length;i++)p[i]/=total;
  return p;
}
function randomGenerator(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function sample(p,random){let value=random();for(let i=0;i<p.length;i++){value-=p[i];if(value<0)return i;}return p.length-1;}
// Checkpoint transport: dequantize the trained tensors once at page load.
function bytes64(text){return Uint8Array.from(atob(text),c=>c.charCodeAt(0));}
function floats64(text){const bytes=bytes64(text),view=new DataView(bytes.buffer),values=new Float32Array(bytes.length/4);for(let i=0;i<values.length;i++)values[i]=view.getFloat32(i*4,true);return values;}
function loadWeights(checkpoint){
  const weights={config:checkpoint.architecture};
  for(const[name,tensor]of Object.entries(checkpoint.tensors)){
    if(tensor.encoding==='float32-le'){weights[name]=floats64(tensor.data);continue;}
    const bytes=bytes64(tensor.data),values=new Int8Array(bytes.buffer),scales=floats64(tensor.scales),width=tensor.shape[1];
    weights[name]=Float32Array.from(values,(q,i)=>q*scales[Math.floor(i/width)]);
  }
  return weights;
}
const tokens = text => text.toLowerCase().match(/[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;:\n-]/g) ?? [];

function decode(pieces) {
  return pieces.join(" ").replace(/ *\n */g, "\n").replace(/ +([.,!?;:])/g, "$1")
    .replace(/(^|\n)[a-z]|\bi\b|\bi'/g, part => part.toUpperCase());
}

function verseFeatures(text, cue) {
  const lines = text.trim().split("\n").map(line => tokens(line).filter(token => /[a-z0-9]/.test(token)));
  const normalized = lines.map(line => line.map(word => word === cue ? "<cue>" : word));
  return {
    lines: normalized.map(line => line.join(" ")).filter(Boolean),
    endings: normalized.map(line => line.at(-1)).filter(Boolean).join(" "),
    hasCue: !cue || lines.some(line => line.includes(cue)),
  };
}

export function createPerformer(checkpoint) {
  const weights = loadWeights(checkpoint);
  const lookup = new Map(checkpoint.vocabulary.map((word, index) => [word, index]));
  const end = lookup.get("<end>");
  const copy = lookup.get("<copy>");
  const newline = lookup.get("\n");
  const promptTokens = checkpoint.vocabulary.flatMap((word, id) =>
    word.startsWith("<") && id !== end && id !== copy ? [id] : []);
  const nonWords = checkpoint.vocabulary.flatMap((word, id) =>
    !/[a-z0-9]/.test(word) ? [id] : []);
  const isChallengeModel = checkpoint.training.performer === "challenge";

  function sampleVerse(promptState, cue, random, temperature) {
    let hidden = promptState;
    const pieces = [];
    let line = 0;
    let atLineStart = true;
    for (let step = 0; step < 48; step++) {
      const logits = modelLogits(weights, hidden);
      // Prompt markers belong to the exchange protocol, outside the verse.
      for (const id of promptTokens) logits[id] = -Infinity;
      if (!cue) logits[copy] = -Infinity;
      // A couplet needs one newline and at least one word on each side.
      if (atLineStart) {
        for (const id of nonWords) logits[id] = -Infinity;
        logits[end] = -Infinity;
        logits[newline] = -Infinity;
      }
      if (line === 0) logits[end] = -Infinity;
      const id = line === 0 && step === 24 ? newline : sample(softmax(logits, temperature), random);
      if (id === end || (id === newline && line === 1)) break;
      if (id === newline) line++;
      atLineStart = id === newline;
      pieces.push(id === copy ? cue : checkpoint.vocabulary[id]);
      hidden = modelStep(weights, hidden, id);
    }
    return decode(pieces);
  }

  return {
    training: checkpoint.training,
    generate({previous = "", cue = "", seed, earlier = [], role}) {
      const prompt = ["<start>"];
      if (isChallengeModel) prompt.push(`<${role}>`);
      if (previous) prompt.push(...tokens(previous), "<cue>", ...tokens(cue));
      prompt.push("<verse>");
      let hidden = new Float32Array(weights.config.hidden);
      for (const word of prompt) {
        hidden = modelStep(weights, hidden, lookup.get(word) ?? lookup.get("<unk>"));
      }
      const random = randomGenerator(seed);
      const used = earlier.map(verse => verseFeatures(verse.text, verse.cue));
      let best = "";
      let bestScore = -1;
      for (const temperature of [0.75, 0.75, 0.85, 0.85, 0.95, 0.95]) {
        const text = sampleVerse(hidden, cue, random, temperature);
        const {lines, endings, hasCue} = verseFeatures(text, cue);
        if (lines.length !== 2) continue;
        // Treat a verse with just the callback swapped as the same verse.
        const fresh = used.every(verse => verse.endings !== endings &&
          lines.every(line => !verse.lines.includes(line)));
        if (fresh && hasCue) return text;
        // A bounded search keeps generation responsive even with a tiny corpus.
        const score = Number(fresh) * 2 + Number(hasCue);
        if (score > bestScore) {
          best = text;
          bestScore = score;
        }
      }
      return best;
    },
  };
}
