"""Fetch a bounded, pinned TinyStories sample and learn a BPE tokenizer.
Raw stories are training inputs only; they are never bundled in the webpage.
"""
import argparse, hashlib, json, re, unicodedata, urllib.request
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import numpy as np

REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
BASE = f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/'
LIMITS = {'train': 24_000_000, 'valid': 1_000_000}
EXPECTED = {'train': '6b612d502aba556b7f42067df03a530a68cdc6cee9495c606ee44487779761b6', 'valid': '7d10e8a610761afce06e10bee271b9e23f4043bd7298d7965a00262c57c818df'}

def normalize(text):
    return unicodedata.normalize('NFKC', text).translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','–':'-','—':'-'}))
def units(text):
    return ' '.join('▁'+word for word in normalize(text).split())

def prepare(root, vocab_size):
    root.mkdir(parents=True, exist_ok=True)
    splits={}
    manifest={'dataset':'roneneldan/TinyStories','revision':REVISION,'license':'CDLA-Sharing-1.0','splits':{}}
    for split,limit in LIMITS.items():
        path=root/(split+'.txt')
        if not path.exists():
            req=urllib.request.Request(BASE+'TinyStories-'+split+'.txt',headers={'Range':f'bytes=0-{limit-1}'})
            with urllib.request.urlopen(req,timeout=60) as response: data=response.read(limit)
            end=data.rfind(b'<|endoftext|>')
            if end<0: raise ValueError('Missing story delimiter')
            path.write_bytes(data[:end+len(b'<|endoftext|>')])
        data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        if digest!=EXPECTED[split]: raise ValueError(f'{split} source checksum mismatch')
        stories=[s.strip() for s in data.decode().split('<|endoftext|>') if s.strip()]
        splits[split]=stories
        manifest['splits'][split]={'bytes':len(data),'sha256':digest,'stories':len(stories)}
    # Keep evaluation documents out of training, even if the upstream splits overlap.
    heldout={hashlib.sha256(units(s).encode()).hexdigest() for s in splits['valid']}
    splits['train']=[s for s in splits['train'] if hashlib.sha256(units(s).encode()).hexdigest() not in heldout]
    tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    alphabet=list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789▁.,!?;:\'"-()')
    tokenizer.train_from_iterator((units(s) for s in splits['train']),trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=3,special_tokens=['[UNK]','[PARA]'],initial_alphabet=alphabet,show_progress=False))
    tokenizer.save(str(root/'tokenizer.json'))
    raw=json.loads(tokenizer.to_str()); vocab=[None]*len(raw['model']['vocab'])
    for word,index in raw['model']['vocab'].items():vocab[index]=word
    token_data={'vocabulary':vocab,'merges':raw['model']['merges'],'unknown':0,'paragraph':1,'normalization':'NFKC, normalized quotation marks, whitespace-separated metaspace units'}
    (root/'tokenizer-compact.json').write_text(json.dumps(token_data,ensure_ascii=False,separators=(',',':')))
    for split,stories in splits.items():
        stream=[]
        for story in stories:
            stream.append(1);stream.extend(tokenizer.encode(units(story)).ids)
        array=np.asarray(stream,dtype=np.int32);np.save(root/(split+'.npy'),array)
        manifest['splits'][split].update({'usedStories':len(stories),'tokens':len(array),'unknownRate':float(np.mean(array==0))})
    manifest['vocabularySize']=len(vocab)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    prepare_descriptions(root,tokenizer)
    print(json.dumps(manifest),flush=True)
def prepare_descriptions(root,tokenizer):
    manifest={}
    for split,file in [('train','descriptions.txt'),('valid','descriptions-valid.txt')]:
        text=Path(__file__).with_name(file).read_text();paragraphs=text.strip().split('\n\n');ids=[]
        for paragraph in paragraphs:ids.extend([1]+tokenizer.encode(units(paragraph)).ids)
        np.save(root/f'descriptions-{split}.npy',np.array(ids,dtype=np.uint16))
        manifest[split]={'file':file,'paragraphs':len(paragraphs),'tokens':len(ids),'sha256':hashlib.sha256(text.encode()).hexdigest()}
    (root/'descriptions-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--vocab',type=int,default=1900)
    args=parser.parse_args();prepare(args.data,args.vocab)
