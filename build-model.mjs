// Verify a trained checkpoint. This file never authors or changes model weights.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
export function verify() {
  const html=readFileSync(new URL('./index.html',import.meta.url),'utf8');
  const encoded=html.match(/<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/)?.[1];
  assert(encoded,'Missing embedded checkpoint');
  const checkpoint=JSON.parse(encoded), {embedding:E,hidden:H,vocabulary:V}=checkpoint.architecture;
  assert.equal(checkpoint.format,4);assert.equal(checkpoint.architecture.type,'GRU');
  assert(Buffer.byteLength(html)<1_000_000,'Entire page must remain below 1 MB');
  assert.equal(checkpoint.tokenizer.vocabulary.length,V);
  const dimensions={embedding:[V,E],input:[3*H,E],recurrent:[3*H,H],inputBias:[3*H],recurrentBias:[3*H],output:[V,H],outputBias:[V]};
  let parameters=0,packedBytes=0;
  for(const[name,shape]of Object.entries(dimensions)){
    const tensor=checkpoint.tensors[name];assert.deepEqual(tensor.shape,shape);
    const count=shape.reduce((a,b)=>a*b,1);parameters+=count;
    const bytes=Buffer.from(tensor.data,'base64');packedBytes+=bytes.length;
    if(shape.length===1){assert.equal(tensor.encoding,'float32-le');assert.equal(bytes.length,count*4);}
    else{assert.equal(tensor.encoding,'int8-row');assert.equal(bytes.length,count);const scales=Buffer.from(tensor.scales,'base64');assert.equal(scales.length,shape[0]*4);packedBytes+=scales.length;}
  }
  assert.equal(parameters,checkpoint.stats.parameters);assert.equal(packedBytes,checkpoint.stats.packedWeightBytes);
  assert.equal(Buffer.byteLength(encoded),checkpoint.stats.checkpointBytes);
  const digest=createHash('sha256').update(encoded).digest('hex');
  const report=JSON.parse(readFileSync(new URL('./training/report.json',import.meta.url)));
  const reference=JSON.parse(readFileSync(new URL('./training/reference.json',import.meta.url)));
  assert.equal(digest,report.checkpointSha256);assert.equal(digest,reference.checkpointSha256);
  assert.equal(checkpoint.training.step,report.bestStep);
  assert(checkpoint.training.quantizedValidationLoss<checkpoint.training.initialValidationLoss/2);
  return{html,encoded,checkpoint,report,reference,parameters,packedBytes,htmlBytes:Buffer.byteLength(html),sha256:digest};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const{parameters,packedBytes,htmlBytes,sha256}=verify();
  console.log(JSON.stringify({parameters,packedBytes,htmlBytes,sha256}));
}
