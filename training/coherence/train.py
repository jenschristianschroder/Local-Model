"""Train a fresh GRU and BPE tokenizer on the local synthetic corpus only."""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('story_gru', ROOT.parent / 'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def units(text):
    return ' '.join('▁' + word for word in text.split())


def encode(tokenizer, paragraphs):
    ids = [1]
    for paragraph in paragraphs:
        ids.extend(tokenizer.encode(units(paragraph)).ids)
        ids.append(1)
    return ids


def batch_tokens(documents):
    batch = torch.full((len(documents), max(map(len, documents))), -100, dtype=torch.long)
    for row, ids in enumerate(documents):
        batch[row, :len(ids)] = torch.tensor(ids)
    return batch


def loss_for(net, batch, weights=None):
    logits, _ = net(batch[:, :-1].clamp_min(0))
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), weight=weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--entity-weight', type=float, default=12)
    args = parser.parse_args()
    if args.entity_weight <= 0:
        parser.error('--entity-weight must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(1311)
    rng = random.Random(1311)
    output = ROOT.parent / 'runs' / 'synthetic-coherence'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for split in ['train', 'valid']:
        digest = hashlib.sha256((ROOT / f'{split}.json').read_bytes()).hexdigest()
        if digest != manifest['splits'][split]['sha256']:
            raise ValueError(f'{split} corpus checksum mismatch; regenerate the corpus manifest')
    stories = {split: json.loads((ROOT / f'{split}.json').read_text()) for split in ['train', 'valid']}
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.train_from_iterator(
        (units(p) for story in stories['train'] for p in story['paragraphs']),
        trainers.BpeTrainer(vocab_size=1024, min_frequency=2,
                            special_tokens=['[UNK]', '[PARA]'],
                            initial_alphabet=list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789▁.,!?;:\'"-()'),
                            show_progress=False))
    tokenizer.save(str(output / 'tokenizer.json'))
    documents = {split: [encode(tokenizer, story['paragraphs']) for story in values]
                 for split, values in stories.items()}
    net = module.StoryGRU(tokenizer.get_vocab_size(), embedding=64, hidden=256)
    weights = torch.ones(tokenizer.get_vocab_size())
    entities = {story[field] for story in stories['train'] for field in ['actor', 'companion', 'object']}
    for entity in entities:
        for token in tokenizer.encode(units(entity)).ids:
            weights[token] = args.entity_weight
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003)
    report = {'seed': 1311, 'initialization': 'random', 'tokenizer': 'learned from new train split only',
              'architecture': {'vocabulary': tokenizer.get_vocab_size(), 'embedding': 64, 'hidden': 256},
              'parameters': sum(p.numel() for p in net.parameters()),
              'entityTokenLossWeight': args.entity_weight,
              'sequenceLength': max(map(len, documents['train'])),
              'source': manifest,
              'torchVersion': torch.__version__, 'history': []}
    best = math.inf
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step:
            net.train()
            for group in optimizer.param_groups:
                group['lr'] = .003 * (.2 + .8 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            batch = batch_tokens(rng.sample(documents['train'], 16))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(net, batch, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        if step % 100 == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                valid = sum(loss_for(net, batch_tokens(documents['valid'][i:i + 16])).item()
                            for i in range(0, len(documents['valid']), 16)) / math.ceil(len(documents['valid']) / 16)
            entry = {'step': step, 'validationLoss': valid, 'seconds': round(time.monotonic() - started, 1)}
            report['history'].append(entry)
            if valid < best:
                best = valid
                report['bestStep'] = step
                torch.save({'state': net.state_dict(), 'report': report}, output / 'best.pt')
            (ROOT / 'training-report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(entry), flush=True)
    print('Checkpoint ' + str(output / 'best.pt'), flush=True)


if __name__ == '__main__':
    main()
