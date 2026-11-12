"""Fine-tune an embedded GRU on ordinary prose. No inference-time genre rules."""
import argparse,base64,hashlib,json,math,re,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train import StoryGRU,evaluate,evaluate_paragraphs,sample
from prepare import units
from tokenizers import Tokenizer,models,pre_tokenizers
from collections import Counter

MAPPING={'embedding':'embedding.weight','input':'gru.weight_ih_l0','recurrent':'gru.weight_hh_l0','inputBias':'gru.bias_ih_l0','recurrentBias':'gru.bias_hh_l0','output':'output.weight','outputBias':'output.bias'}
def read_model(path):
    encoded=re.search(r'<script id="model-data" type="application/json">([\s\S]*?)</script>',path.read_text())[1]
    bundle=json.loads(encoded);state={}
    for name,t in bundle['tensors'].items():
        raw=base64.b64decode(t['data'])
        if t['encoding']=='int8-row':
            q=np.frombuffer(raw,dtype=np.int8).reshape(t['shape']);scales=np.frombuffer(base64.b64decode(t['scales']),dtype='<f4');a=q.astype(np.float32)*scales[:,None]
        else:a=np.frombuffer(raw,dtype='<f4').copy()
        state[MAPPING[name]]=torch.from_numpy(a.copy())
    return bundle,state,hashlib.sha256(encoded.encode()).hexdigest()

def prepare_corpus(data,tokenizer):
    manifest={};streams={}
    for split,name in [('train','christmas.txt'),('valid','christmas-valid.txt')]:
        text=Path(__file__).with_name(name).read_text();documents=text.strip().split('\n\n');ids=[];lengths=[]
        for document in documents:
            tokens=[1]+tokenizer.encode(units(document)).ids;ids.extend(tokens);lengths.append(len(tokens))
        streams[split]=np.array(ids,dtype=np.int64)
        np.save(data/f'christmas-{split}.npy',streams[split])
        manifest[split]={'file':name,'paragraphs':len(documents),'tokens':len(ids),'sha256':hashlib.sha256(text.encode()).hexdigest(),'maxTokens':max(lengths),'unknownTokens':ids.count(0)}
    train_docs=set(Path(__file__).with_name('christmas.txt').read_text().strip().split('\n\n'))
    valid_docs=set(Path(__file__).with_name('christmas-valid.txt').read_text().strip().split('\n\n'))
    if train_docs&valid_docs:raise ValueError('Training and validation documents overlap')
    (data/'christmas-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return streams,manifest

def extend_vocabulary(bundle,state,tokenizer,count,data):
    """Learn extra merge ranks from training text only; initialize new rows compositionally."""
    words=Counter(units(Path(__file__).with_name('christmas.txt').read_text()).split())
    vocabulary=list(bundle['tokenizer']['vocabulary']);merges=[tuple(vocabulary[i] for i in row[:2]) for row in bundle['tokenizer']['merges']]
    rows={word:tokenizer.encode(word).ids for word in words};extensions=[]
    for _ in range(count):
        pairs=Counter()
        for word,ids in rows.items():
            for pair in zip(ids,ids[1:]):pairs[pair]+=words[word]
        candidates=[(freq,a,b) for (a,b),freq in pairs.items() if vocabulary[a]+vocabulary[b] not in vocabulary]
        freq,a,b=max(candidates,key=lambda item:(item[0],-item[1],-item[2]));out=len(vocabulary)
        vocabulary.append(vocabulary[a]+vocabulary[b]);merges.append((vocabulary[a],vocabulary[b]));extensions.append({'a':a,'b':b,'id':out,'frequency':freq,'piece':vocabulary[-1]})
        for key in ['embedding.weight','output.weight']:
            state[key]=torch.cat([state[key],((state[key][a]+state[key][b])*.5)[None,:]],dim=0)
        bias=state['output.bias'];state['output.bias']=torch.cat([bias,((bias[a]+bias[b])*.5).reshape(1)])
        for word,ids in list(rows.items()):
            result=[];i=0
            while i<len(ids):
                if i+1<len(ids) and ids[i]==a and ids[i+1]==b:result.append(out);i+=2
                else:result.append(ids[i]);i+=1
            rows[word]=result
    tokenizer=Tokenizer(models.BPE(vocab={word:i for i,word in enumerate(vocabulary)},merges=merges,unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.WhitespaceSplit();tokenizer.add_special_tokens(['[UNK]','[PARA]']);tokenizer.save(str(data/'tokenizer.json'))
    (data/'tokenizer-compact.json').write_text(json.dumps({'vocabulary':vocabulary,'merges':merges,'unknown':0,'paragraph':1},ensure_ascii=False,separators=(',',':')))
    manifest=json.loads((data/'manifest.json').read_text())
    heldout=set(units(s.strip()) for s in (data/'valid.txt').read_text().split('<|endoftext|>') if s.strip())
    for split in ['train','valid']:
        docs=[units(s.strip()) for s in (data/f'{split}.txt').read_text().split('<|endoftext|>') if s.strip()]
        if split=='train':docs=[doc for doc in docs if doc not in heldout]
        encoded=tokenizer.encode_batch(docs);stream=[]
        for encoded_doc in encoded:stream.extend([1]+encoded_doc.ids)
        np.save(data/f'{split}.npy',np.array(stream,dtype=np.int32));manifest['splits'][split]['tokens']=len(stream)
    manifest['vocabularySize']=len(vocabulary);(data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from prepare import prepare_descriptions
    prepare_descriptions(data,tokenizer)
    return tokenizer,extensions

def adapt(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    bundle,state,base_sha=read_model(args.base_html)
    tok=Tokenizer.from_file(str(args.data/'tokenizer.json'))
    assert [tok.id_to_token(i) for i in range(tok.get_vocab_size())]==bundle['tokenizer']['vocabulary']
    extensions=[]
    if args.merges:tok,extensions=extend_vocabulary(bundle,state,tok,args.merges,args.data)
    streams,manifest=prepare_corpus(args.data,tok);base=np.load(args.data/'train.npy');valid=np.load(args.data/'valid.npy')
    V=tok.get_vocab_size();net=StoryGRU(V,bundle['architecture']['embedding'],bundle['architecture']['hidden']);net.load_state_dict(state)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01);rng=np.random.default_rng(args.seed)
    report=json.loads(args.base_report.read_text());report.pop('export',None);report.pop('checkpointSha256',None)
    report['vocabulary']=V;report['parameters']=sum(p.numel() for p in net.parameters());report['fineTuneTrainingTokens']=len(base);report['fineTuneValidationTokens']=len(valid)
    start=bundle['training']['step'];baseline=evaluate_paragraphs(net,streams['valid'])
    holiday={'name':'Absurd Christmas stories','corpus':manifest,'initialValidationLoss':baseline,'baseCheckpointSha256':base_sha,'seed':args.seed,'batchProbability':args.fraction,'sequenceLength':args.length,'batchSize':args.batch,'baseLearningRate':args.lr,'history':[],'vocabularyExtensions':extensions,'selectionMetric':'0.85 * full Christmas paragraph loss + 0.15 * TinyStories window loss'}
    report['holiday']=holiday;report['selectionMetric']=holiday['selectionMetric'];report['phases'].append({'name':'Christmas fine-tuning','startStep':start,'targetSteps':start+args.steps,'batchSize':args.batch,'sequenceLength':args.length,'baseLearningRate':args.lr,'extraBatchProbability':args.fraction,'restartSchedule':True,'wholeParagraphs':True})
    print(json.dumps({'initialChristmasValidationLoss':baseline,'corpus':manifest}),flush=True)
    args.output.mkdir(parents=True,exist_ok=True);starts=np.flatnonzero(streams['train']==1);documents=[]
    for i,a in enumerate(starts):
        b=starts[i+1] if i+1<len(starts) else len(streams['train']);documents.append(np.r_[streams['train'][a:b],1])
    best=math.inf;running=0;began=time.perf_counter()
    for local_step in range(1,args.steps+1):
        if rng.random()<args.fraction:
            array=np.full((args.batch,args.length+1),-100,dtype=np.int64)
            for row,which in enumerate(rng.integers(0,len(documents),size=args.batch)):
                tokens=documents[which][:args.length+1];array[row,:len(tokens)]=tokens
        else:
            positions=rng.integers(0,len(base)-args.length-1,size=args.batch);array=base[positions[:,None]+np.arange(args.length+1)].astype(np.int64)
        batch=torch.from_numpy(array);inputs=batch[:,:-1].clone();inputs[inputs<0]=0;inputs[torch.rand(inputs.shape)<.003]=0
        lr=args.lr*(.15+.85*.5*(1+math.cos(math.pi*local_step/args.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);logits,_=net(inputs);loss=nn.functional.cross_entropy(logits.reshape(-1,V),batch[:,1:].reshape(-1));loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();running+=loss.item()
        if local_step%100==0 or local_step==args.steps:
            xmas=evaluate_paragraphs(net,streams['valid']);story=evaluate(net,valid,batches=8);score=.85*xmas+.15*story
            entry={'step':start+local_step,'adaptationStep':local_step,'trainLoss':running/min(100,local_step),'validationLoss':story,'christmasValidationLoss':xmas,'selectionScore':score,'learningRate':lr,'seconds':round(time.perf_counter()-began,2)}
            holiday['history'].append(entry);report['history'].append(entry);report['completedSteps']=start+local_step;running=0
            if score<best:
                best=score;report['bestStep']=start+local_step;report['bestValidationLoss']=story;report['bestSelectionScore']=score;holiday['bestValidationLoss']=xmas;holiday['bestAdaptationStep']=local_step
                torch.save({'state':net.state_dict(),'report':report},args.output/'best.pt')
            torch.save({'state':net.state_dict(),'report':report,'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,'torchRng':torch.get_rng_state(),'step':start+local_step},args.output/'latest.pt')
            (args.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(entry),flush=True)
            if local_step%400==0 or local_step==args.steps:print('SAMPLE Santa'+sample(net,args.data/'tokenizer.json','Santa',110,42,.65),flush=True)
    print(json.dumps({'bestStep':report['bestStep'],'bestChristmasLoss':holiday['bestValidationLoss']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-html',type=Path,required=True);p.add_argument('--base-report',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1400);p.add_argument('--length',type=int,default=160);p.add_argument('--batch',type=int,default=24);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=1225);p.add_argument('--lr',type=float,default=.001);p.add_argument('--fraction',type=float,default=.8)
    p.add_argument('--merges',type=int,default=0,help='Extra BPE merges learned from training prose only')
    adapt(p.parse_args())
