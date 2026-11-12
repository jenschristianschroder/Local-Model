"""Train a small GRU language model from random initialization; no grammar code.
Requires the prepared TinyStories streams. All trainable parameters are learned.
"""
import argparse, json, math, time, hashlib, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

class StoryGRU(nn.Module):
    def __init__(self,vocabulary,embedding=64,hidden=192):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,embedding)
        self.gru=nn.GRU(embedding,hidden,batch_first=True)
        self.output=nn.Linear(hidden,vocabulary)
        self.dropout=nn.Dropout(.08)
        nn.init.normal_(self.embedding.weight,mean=0,std=.12)
    def forward(self,tokens,hidden=None):
        h,hidden=self.gru(self.dropout(self.embedding(tokens)),hidden)
        return self.output(self.dropout(h)),hidden

def evaluate(net,stream,seed=2026,batches=12,batch_size=32,length=64):
    random_state=np.random.default_rng(seed);total=0
    net.eval()
    with torch.no_grad():
        for _ in range(batches):
            starts=random_state.integers(0,len(stream)-length-1,size=batch_size)
            indexes=starts[:,None]+np.arange(length+1)
            batch=torch.from_numpy(stream[indexes].astype(np.int64))
            logits,_=net(batch[:,:-1]);total+=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1)).item()
    net.train();return total/batches

def evaluate_paragraphs(net,stream):
    """Token-weighted loss with a fresh state for each complete held-out document."""
    starts=np.flatnonzero(stream==1);total=0;count=0;net.eval()
    with torch.no_grad():
        for i,start in enumerate(starts):
            end=starts[i+1] if i+1<len(starts) else len(stream)
            tokens=torch.from_numpy(stream[start:end].astype(np.int64))[None,:]
            logits,_=net(tokens[:,:-1])
            total+=nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1),reduction='sum').item()
            count+=tokens.shape[1]-1
    net.train();return total/count

def sample(net,tokenizer,prompt,count=120,seed=42,temperature=.7):
    from tokenizers import Tokenizer
    from prepare import units
    tokenizer=Tokenizer.from_file(str(tokenizer));tokens=[1]+tokenizer.encode(units(prompt)).ids
    gen=torch.Generator().manual_seed(seed);net.eval()
    with torch.no_grad():
        logits,h=net(torch.tensor([tokens]))
        result=[]
        for _ in range(count):
            nxt=torch.multinomial(torch.softmax(logits[0,-1]/temperature,dim=-1),1,generator=gen)
            result.append(int(nxt));logits,h=net(nxt.reshape(1,1),h)
    net.train()
    return ''.join(tokenizer.id_to_token(t) if t!=1 else '\n\n' for t in result).replace('▁',' ')

def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    data=args.data;train_stream=np.load(data/'train.npy');valid_stream=np.load(data/'valid.npy')
    extra=np.load(args.extra) if args.extra else None
    extra_valid=np.load(args.extra_valid) if args.extra_valid else None
    extra_starts=np.flatnonzero(extra==1) if extra is not None else []
    if extra is not None and extra_valid is None:raise ValueError('--extra-valid is required with --extra')
    tokenizer=json.loads((data/'tokenizer-compact.json').read_text());V=len(tokenizer['vocabulary'])
    net=StoryGRU(V,args.embedding,args.hidden)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    report={'architecture':'GRU','hidden':args.hidden,'embedding':args.embedding,'vocabulary':V,'parameters':sum(p.numel() for p in net.parameters()),'seed':args.seed,'trainingTokens':len(train_stream),'validationTokens':len(valid_stream),'batchSize':args.batch,'sequenceLength':args.length,'history':[],'source':json.loads((data/'manifest.json').read_text()),'torchVersion':torch.__version__}
    random_state=np.random.default_rng(args.seed);best=math.inf;start_step=0
    if args.resume and args.resume.exists():
        saved=torch.load(args.resume,weights_only=False,map_location='cpu');net.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);report=saved['report'];start_step=saved['step'];random_state.bit_generator.state=saved['rng'];best=report.get('bestValidationLoss',math.inf)
        if 'torchRng' in saved and not args.reset_torch_rng:torch.set_rng_state(saved['torchRng'])
    else:
        baseline=evaluate(net,valid_stream,batches=8);report['initialValidationLoss']=baseline
        print(json.dumps({'step':0,'validationLoss':baseline,'perplexity':math.exp(baseline),'parameters':report['parameters']}),flush=True)
    if extra is not None:
        report['extraCorpus']=json.loads((data/'descriptions-manifest.json').read_text())
        report['selectionMetric']='0.7 * TinyStories loss + 0.3 * held-out descriptions loss'
        best=math.inf
    report.setdefault('phases',[]).append({'startStep':start_step,'targetSteps':args.steps,'batchSize':args.batch,'sequenceLength':args.length,'baseLearningRate':args.lr,'extraBatchProbability':args.extra_fraction if extra is not None else 0,'restartSchedule':args.restart_schedule,'fullDescriptionParagraphs':args.extra_paragraphs})
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();running=0
    for step in range(start_step+1,args.steps+1):
        use_extra=extra is not None and random_state.random()<args.extra_fraction
        stream=extra if use_extra else train_stream
        if use_extra and args.extra_paragraphs:
            chosen=random_state.integers(0,len(extra_starts),size=args.batch)
            array=np.full((args.batch,args.length+1),-100,dtype=np.int64)
            for row,doc in enumerate(chosen):
                start=extra_starts[doc];end=extra_starts[doc+1]+1 if doc+1<len(extra_starts) else len(extra)
                tokens=extra[start:min(end,start+args.length+1)];array[row,:len(tokens)]=tokens
            batch=torch.from_numpy(array)
        else:
            starts=random_state.integers(0,len(stream)-args.length-1,size=args.batch)
            indexes=starts[:,None]+np.arange(args.length+1)
            batch=torch.from_numpy(stream[indexes].astype(np.int64))
        inputs=batch[:,:-1].clone()
        inputs[inputs<0]=0
        # Input corruption trains an unknown embedding. Targets are untouched.
        inputs[torch.rand(inputs.shape)<.003]=0
        progress=(step-start_step)/(args.steps-start_step) if args.restart_schedule else step/args.steps
        lr=args.lr*(.2+.8*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        logits,_=net(inputs)
        loss=nn.functional.cross_entropy(logits.reshape(-1,V),batch[:,1:].reshape(-1))
        loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);optimizer.step();running+=loss.item()
        if step%args.evaluate_every==0 or step==args.steps:
            val=evaluate(net,valid_stream)
            entry={'step':step,'trainLoss':running/min(args.evaluate_every,step-start_step),'validationLoss':val,'perplexity':math.exp(val),'seconds':round(time.perf_counter()-started,2),'learningRate':lr}
            score=val
            if extra_valid is not None:
                entry['descriptionValidationLoss']=evaluate(net,extra_valid,batches=8)
                score=.7*val+.3*entry['descriptionValidationLoss']
                entry['selectionScore']=score
            report['history'].append(entry);running=0
            report['completedSteps']=step
            if score<best:
                best=score;report['bestValidationLoss']=val;report['bestSelectionScore']=score;report['bestStep']=step
                torch.save({'state':net.state_dict(),'report':report},args.output/'best.pt')
            report['completedSteps']=step
            torch.save({'state':net.state_dict(),'optimizer':optimizer.state_dict(),'report':report,'step':step,'rng':random_state.bit_generator.state,'torchRng':torch.get_rng_state()},args.output/'latest.pt')
            (args.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(entry),flush=True)
            if step%1000==0 or step==args.steps:
                print('SAMPLE '+sample(net,data/'tokenizer.json','The little robot',90,42),flush=True)
    print('BEST '+str(report['bestStep'])+' '+str(report['bestValidationLoss']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int,default=32);p.add_argument('--length',type=int,default=64)
    p.add_argument('--hidden',type=int,default=192);p.add_argument('--embedding',type=int,default=64);p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=2);p.add_argument('--lr',type=float,default=.003);p.add_argument('--evaluate-every',type=int,default=250)
    p.add_argument('--extra',type=Path);p.add_argument('--extra-valid',type=Path);p.add_argument('--extra-fraction',type=float,default=.3);p.add_argument('--restart-schedule',action='store_true')
    p.add_argument('--reset-torch-rng',action='store_true',help='Repeat the seed-based phase initialization used by the released training run')
    p.add_argument('--extra-paragraphs',action='store_true')
    train(p.parse_args())
