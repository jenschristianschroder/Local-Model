"""Offline continuation training with sequence unlikelihood. No browser rules.

Positive examples teach paragraph transitions; sampled repeated spans supply a
negative training signal. Both losses update the same GRU parameter tensors.
"""
import argparse, copy, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from adapt import read_model
from prepare import units
from train import StoryGRU, evaluate, evaluate_paragraphs


def read_documents(path, tokenizer, separated=False):
    text=path.read_text()
    chunks=text.strip().split('<|endoftext|>' if separated else '\n\n')
    docs=[]
    for chunk in chunks:
        ids=[1]
        for paragraph in chunk.strip().split('\n\n'):
            if paragraph.strip():ids.extend(tokenizer.encode(units(paragraph)).ids);ids.append(1)
        if len(ids)>2:docs.append(ids)
    return docs,{'file':path.name,'documents':len(docs),'paragraphs':sum(d.count(1)-1 for d in docs),'tokens':sum(len(d) for d in docs),'sha256':hashlib.sha256(text.encode()).hexdigest()}


def repetition_mask(tokens, prefix, n=8, target='span'):
    """Mark tokens in repeated n-grams for offline negative-example training."""
    mask=torch.zeros_like(tokens,dtype=torch.bool)
    for row,ids in enumerate(tokens.tolist()):
        seen=set()
        for end in range(n,len(ids)+1):
            gram=tuple(ids[end-n:end])
            if gram in seen and end>prefix:
                if target=='end':mask[row,end-1]=True
                else:mask[row,max(prefix,end-n):end]=True
            seen.add(gram)
    return mask


def rollout(net, prefixes, count, temperature, generator):
    net.eval()
    with torch.no_grad():
        logits,h=net(prefixes);parts=[prefixes]
        for _ in range(count):
            z=logits[:,-1]
            nxt=z.argmax(-1,keepdim=True) if temperature==0 else torch.multinomial(torch.softmax(z/temperature,-1),1,generator=generator)
            parts.append(nxt);logits,h=net(nxt,h)
    return torch.cat(parts,1)


def word_repetition(ids, tokenizer, n=8):
    text=''.join('\n\n' if x==1 else tokenizer.id_to_token(x) for x in ids).replace('▁',' ')
    words=text.split();grams=[tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return 1-len(set(grams))/max(1,len(grams)),text


def quantized_model(net):
    """Match export.py's row-wise int8 representation for offline rollouts."""
    result=copy.deepcopy(net)
    with torch.no_grad():
        for p in result.parameters():
            if p.ndim!=2:continue
            a=p.numpy();scale=np.maximum(np.max(np.abs(a),axis=1)/127,1e-12).astype(np.float32)
            q=np.clip(np.rint(a/scale[:,None]),-127,127).astype(np.int8)
            p.copy_(torch.from_numpy(q.astype(np.float32)*scale[:,None]))
    return result


def dev_repetition(net, tokenizer, quantized=False):
    if quantized:net=quantized_model(net)
    values=[];samples=[]
    for prompt in ['Santa opened the fridge','The Christmas pudding','A reindeer','On Christmas Eve']:
        prefix=torch.tensor([[1]+tokenizer.encode(units(prompt)).ids])
        ids=rollout(net,prefix,1024 if quantized else 768,0,None)[0,prefix.shape[1]:].tolist()
        repeat,text=word_repetition(ids,tokenizer);values.append(repeat);samples.append({'prompt':prompt,'text':prompt+text,'repeated8':repeat})
    return float(np.mean(values)),samples


def document_loss(net, docs):
    net.eval();total=count=0
    with torch.no_grad():
        for ids in docs:
            x=torch.tensor([ids]);z,_=net(x[:,:-1])
            total+=nn.functional.cross_entropy(z.flatten(0,1),x[:,1:].flatten(),reduction='sum').item();count+=len(ids)-1
    return total/count


def repair(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);generator=torch.Generator().manual_seed(args.seed+1)
    bundle,state,sha=read_model(args.base_html);tok=Tokenizer.from_file(str(args.data/'tokenizer.json'))
    assert [tok.id_to_token(i) for i in range(tok.get_vocab_size())]==bundle['tokenizer']['vocabulary']
    root=Path(__file__).parent
    old,_=read_documents(root/'christmas.txt',tok)
    long_docs,train_info=read_documents(root/'christmas-long.txt',tok,True)
    long_valid,valid_info=read_documents(root/'christmas-long-valid.txt',tok,True)
    assert not set(map(tuple,long_docs))&set(map(tuple,long_valid))
    base=np.load(args.data/'train.npy');valid=np.load(args.data/'valid.npy');holiday_valid=np.load(args.data/'christmas-valid.npy')
    net=StoryGRU(bundle['architecture']['vocabulary'],bundle['architecture']['embedding'],bundle['architecture']['hidden']);net.load_state_dict(state)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    report=json.loads(args.base_report.read_text());report.pop('export',None);report.pop('checkpointSha256',None)
    start=bundle['training']['step'];initial_repeat,_=dev_repetition(net,tok,args.quantized_rollouts)
    evidence={'baseCheckpointSha256':sha,'seed':args.seed,'steps':args.steps,'learningRate':args.lr,'batchSize':args.batch,'sequenceLength':args.length,'longStoryProbability':(1-args.base_fraction)*args.long_fraction,'shortChristmasProbability':(1-args.base_fraction)*(1-args.long_fraction),'baseStoryProbability':args.base_fraction,'unlikelihoodEvery':4,'unlikelihoodWeight':args.alpha,'negativeNgramTokens':args.ngram,'rolloutTokens':args.rollout,'initialRepeated8':initial_repeat,'corpus':{'train':train_info,'valid':valid_info},'initialLongValidationLoss':document_loss(net,long_valid),'history':[]}
    evidence['negativeTarget']=args.repeat_target;evidence['quantizedRollouts']=args.quantized_rollouts;evidence['developmentTokens']=1024 if args.quantized_rollouts else 768;report['repair']=evidence
    report['phases'].append({'name':'Paragraph continuation and sequence unlikelihood','startStep':start,'targetSteps':start+args.steps})
    args.output.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'initial':evidence}),flush=True)
    prefix_pool=[]
    for doc in old+long_docs:
        for offset in [i for i,x in enumerate(doc[:-12]) if x==1]:prefix_pool.append(doc[offset:offset+12])
    began=time.perf_counter();running_ce=running_ul=0
    for step in range(1,args.steps+1):
        array=[]
        for _ in range(args.batch):
            if rng.random()<args.base_fraction:
                at=int(rng.integers(0,len(base)-args.length-1));ids=base[at:at+args.length+1].tolist()
            else:
                ids=[1]
                while len(ids)<args.length+1:
                    pool=long_docs if rng.random()<args.long_fraction else old
                    ids.extend(pool[int(rng.integers(len(pool)))][1:])
            array.append(ids[:args.length+1])
        x=torch.tensor(array);net.train();inputs=x[:,:-1].clone();inputs[torch.rand(inputs.shape)<.003]=0
        optimizer.zero_grad(set_to_none=True);z,_=net(inputs);ce=nn.functional.cross_entropy(z.flatten(0,1),x[:,1:].flatten());loss=ce
        ul=ce.new_tensor(0.)
        if step%4==0:
            prefixes=torch.tensor([prefix_pool[int(rng.integers(len(prefix_pool)))] for _ in range(4)])
            tokens=rollout(quantized_model(net) if args.quantized_rollouts else net,prefixes,args.rollout,0 if rng.random()<.8 else .55,generator)
            mask=repetition_mask(tokens,12,args.ngram,args.repeat_target)[:,1:]
            if mask.any():
                net.eval();z,_=net(tokens[:,:-1]);p=z.log_softmax(-1).gather(-1,tokens[:,1:,None]).squeeze(-1).exp()
                ul=-torch.log1p(-p.clamp(max=1-1e-6))[mask].mean();loss=loss+args.alpha*ul
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1)
        lr=args.lr*(.2+.8*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();running_ce+=ce.item();running_ul+=ul.item()
        if step%75==0 or step==args.steps:
            holiday=evaluate_paragraphs(net,holiday_valid);long_loss=document_loss(net,long_valid);story=evaluate(net,valid,batches=8)
            repeat,samples=dev_repetition(net,tok,args.quantized_rollouts)
            entry={'step':start+step,'repairStep':step,'adaptationStep':report['holiday']['selectedAdaptationStep']+step,'trainLoss':running_ce/min(75,step),'unlikelihoodLossPerStep':running_ul/min(75,step),'validationLoss':story,'christmasValidationLoss':holiday,'longValidationLoss':long_loss,'greedyRepeated8':repeat,'seconds':round(time.perf_counter()-began,2)}
            running_ce=running_ul=0;evidence['history'].append(entry);report['history'].append(entry);report['completedSteps']=start+step
            torch.save({'state':net.state_dict(),'report':report,'step':start+step},args.output/f'step-{step}.pt')
            (args.output/f'samples-{step}.json').write_text(json.dumps(samples,indent=2)+'\n')
            (args.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(entry),flush=True)
    print('Training complete; select and export a measured checkpoint.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-html',type=Path,required=True);p.add_argument('--base-report',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=450);p.add_argument('--length',type=int,default=320);p.add_argument('--batch',type=int,default=12);p.add_argument('--lr',type=float,default=.00025);p.add_argument('--alpha',type=float,default=1);p.add_argument('--seed',type=int,default=923)
    p.add_argument('--base-fraction',type=float,default=.1);p.add_argument('--long-fraction',type=float,default=2/3,help='Long-story fraction conditional on a Christmas batch row');p.add_argument('--ngram',type=int,default=8);p.add_argument('--rollout',type=int,default=512)
    p.add_argument('--repeat-target',choices=['span','end'],default='span')
    p.add_argument('--quantized-rollouts',action='store_true')
    repair(p.parse_args())
