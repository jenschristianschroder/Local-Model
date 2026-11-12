"""Quantize learned tensors, evaluate the exported model, and embed it in HTML."""
import argparse, base64, hashlib, json, math, re
from pathlib import Path
import numpy as np
import torch
from train import StoryGRU, evaluate, evaluate_paragraphs
from prepare import units
from tokenizers import Tokenizer

def b64(a):return base64.b64encode(a.tobytes()).decode()
def pack_tensor(tensor):
    a=tensor.detach().cpu().numpy().astype(np.float32)
    if a.ndim==1:return {'shape':list(a.shape),'encoding':'float32-le','data':b64(a.astype('<f4'))},a
    scale=np.maximum(np.max(np.abs(a),axis=1)/127,1e-12).astype('<f4')
    q=np.clip(np.rint(a/scale[:,None]),-127,127).astype(np.int8)
    return {'shape':list(a.shape),'encoding':'int8-row','data':b64(q),'scales':b64(scale)},q.astype(np.float32)*scale[:,None]

def export(args):
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,weights_only=False,map_location='cpu');report=saved['report']
    if args.selection=='latest':
        selected=saved['step'];entry=next(e for e in reversed(report['history']) if e['step']==selected)
        report['releaseSelection']={'strategy':'Final fine-tuned checkpoint selected after qualitative sample review','selectedStep':selected,'validationBestStep':report['bestStep'],'validationBestLoss':report['holiday']['bestValidationLoss'],'selectedChristmasValidationLoss':entry['christmasValidationLoss']}
        report['bestStep']=selected;report['bestValidationLoss']=entry['validationLoss']
        report['holiday']['selectedAdaptationStep']=entry['adaptationStep']
    elif 'holiday' in report:report['holiday']['selectedAdaptationStep']=report['holiday']['bestAdaptationStep']
    net=StoryGRU(report['vocabulary'],report['embedding'],report['hidden']);net.load_state_dict(saved['state']);net.eval()
    mapping={'embedding':'embedding.weight','input':'gru.weight_ih_l0','recurrent':'gru.weight_hh_l0','inputBias':'gru.bias_ih_l0','recurrentBias':'gru.bias_hh_l0','output':'output.weight','outputBias':'output.bias'}
    tensors={};quantized={};packed_bytes=0
    for name,key in mapping.items():
        tensors[name],a=pack_tensor(saved['state'][key]);quantized[key]=torch.from_numpy(a)
        packed_bytes+=len(base64.b64decode(tensors[name]['data']))+len(base64.b64decode(tensors[name].get('scales','')))
    valid=np.load(args.data/'valid.npy');float_loss=evaluate(net,valid,batches=24)
    holiday_float=evaluate_paragraphs(net,np.load(args.data/'christmas-valid.npy')) if 'holiday' in report else None
    net.load_state_dict(quantized);quant_loss=evaluate(net,valid,batches=24)
    description_loss=evaluate(net,np.load(args.data/'descriptions-valid.npy'),batches=8) if 'extraCorpus' in report else None
    paragraph_loss=evaluate_paragraphs(net,np.load(args.data/'descriptions-valid.npy')) if 'extraCorpus' in report else None
    holiday_quant=evaluate_paragraphs(net,np.load(args.data/'christmas-valid.npy')) if 'holiday' in report else None
    compact=json.loads((args.data/'tokenizer-compact.json').read_text());vocabulary=compact['vocabulary'];lookup={t:i for i,t in enumerate(vocabulary)}
    merges=[[lookup[a],lookup[b],lookup[a+b]] for a,b in compact['merges']]
    training={'dataset':'TinyStories','source':'https://huggingface.co/datasets/roneneldan/TinyStories','license':'CDLA-Sharing-1.0','trainStories':report['source']['splits']['train']['usedStories'],'validationStories':report['source']['splits']['valid']['usedStories'],'trainingTokens':report['trainingTokens'],'validationTokens':report['validationTokens'],'step':report['bestStep'],'seed':report['seed'],'initialValidationLoss':report['initialValidationLoss'],'floatValidationLoss':float_loss,'quantizedValidationLoss':quant_loss,'perplexity':math.exp(quant_loss),'history':[{k:e[k] for k in ['step','trainLoss','validationLoss']} for e in report['history']]}
    if 'extraCorpus' in report:training['descriptions']={**report['extraCorpus'],'validationLoss':description_loss,'fullParagraphLoss':paragraph_loss,'batchProbability':next((phase['extraBatchProbability'] for phase in report['phases'] if phase.get('extraBatchProbability',0)>0 and phase.get('name')!='Christmas fine-tuning'),.3)}
    if 'holiday' in report:
        training.pop('history',None)
        h=report['holiday'];training['holiday']={k:h[k] for k in ['name','corpus','initialValidationLoss','seed','batchProbability','bestAdaptationStep','selectedAdaptationStep','selectionMetric']}
        training['holiday'].update({'floatValidationLoss':holiday_float,'quantizedValidationLoss':holiday_quant,'history':[{k:e[k] for k in ['adaptationStep','christmasValidationLoss']} for e in h['history']]})
    payload={'format':4,'architecture':{'type':'GRU','embedding':report['embedding'],'hidden':report['hidden'],'vocabulary':report['vocabulary'],'gateOrder':'reset,update,new'},'tensors':tensors,'tokenizer':{'vocabulary':vocabulary,'merges':merges,'unknown':0,'paragraph':1},'training':training,'stats':{'parameters':report['parameters'],'packedWeightBytes':packed_bytes},'edits':{'version':4,'weights':{}}}
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    payload['stats']['checkpointBytes']=len(text.encode())
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    # Stabilize the self-describing byte count.
    payload['stats']['checkpointBytes']=len(text.encode());text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'checkpoint.json').write_text(text)
    report.update({'export':{'packedWeightBytes':packed_bytes,'checkpointBytes':len(text.encode()),'floatValidationLoss':float_loss,'quantizedValidationLoss':quant_loss,'floatPerplexity':math.exp(float_loss),'quantizedPerplexity':math.exp(quant_loss),'descriptionValidationLoss':description_loss,'fullParagraphValidationLoss':paragraph_loss,'christmasFloatValidationLoss':holiday_float,'christmasQuantizedValidationLoss':holiday_quant},'checkpointSha256':hashlib.sha256(text.encode()).hexdigest()})
    (args.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Independent PyTorch reference vectors for browser-kernel parity checks.
    tokenizer=Tokenizer.from_file(str(args.data/'tokenizer.json'))
    prompts=['Santa opened the fridge','The Christmas pudding','A reindeer','On Christmas Eve','The snowman borrowed a hat.','Merry Christmas!','The little robot','the little robot','The quiet forest','A curious cat','The distant planet','The robots were','The robot was','The cats were','The cat was','Once upon a time','A telescope watched the stars.','"Hello," said Lily.','A blue-green butterfly!','Unknownword123','A curious cat wakes quietly. The little robot','She found a small key and opened the door.']
    references=[];net.eval()
    with torch.no_grad():
        for prompt in prompts:
            ids=[1]+tokenizer.encode(units(prompt)).ids
            logits,h=net(torch.tensor([ids]));z=logits[0,-1].numpy();hidden=h[0,0].numpy()
            references.append({'prompt':prompt,'ids':ids,'hidden':hidden.tolist(),'logits':z.tolist()})
    (args.output/'reference.json').write_text(json.dumps({'checkpointSha256':report['checkpointSha256'],'cases':references},separators=(',',':')))
    html=args.html.read_text();pattern=r'(<script id="model-data" type="application/json">)[\s\S]*?(</script>)'
    if not re.search(pattern,html):raise ValueError('Missing model-data block')
    html=re.sub(pattern,lambda m:m[1]+text+m[2],html)
    if len(html.encode())>1_000_000:raise ValueError(f'HTML exceeds 1 MB: {len(html.encode())} bytes')
    args.html.write_text(html)
    print(json.dumps({'parameters':report['parameters'],'packedWeightBytes':packed_bytes,'checkpointBytes':len(text.encode()),'htmlBytes':len(html.encode()),'floatValidationLoss':float_loss,'quantizedValidationLoss':quant_loss,'quantizedPerplexity':math.exp(quant_loss),'bestStep':report['bestStep']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--html',type=Path,required=True);p.add_argument('--selection',choices=['best','latest'],default='best');export(p.parse_args())
