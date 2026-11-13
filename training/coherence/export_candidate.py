"""Export the trained candidate using the existing tensor quantizer."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import torch
from export import pack_tensor
from train import StoryGRU
from tokenizers import Tokenizer


def main():
    output = ROOT.parent / 'runs' / 'synthetic-coherence'
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    report = saved['report']
    tokenizer_text = (output / 'tokenizer.json').read_text()
    raw = json.loads(tokenizer_text)['model']
    lookup = raw['vocab']
    vocabulary = sorted(lookup, key=lookup.get)
    mapping = {'embedding': 'embedding.weight', 'input': 'gru.weight_ih_l0',
               'recurrent': 'gru.weight_hh_l0', 'inputBias': 'gru.bias_ih_l0',
               'recurrentBias': 'gru.bias_hh_l0', 'output': 'output.weight',
               'outputBias': 'output.bias'}
    tensors = {}
    state = {}
    for name, key in mapping.items():
        tensors[name], array = pack_tensor(saved['state'][key])
        state[key] = torch.from_numpy(array)
    payload = {'format': 4, 'architecture': {'type': 'GRU', **report['architecture'],
               'gateOrder': 'reset,update,new'}, 'tensors': tensors,
               'tokenizer': {'vocabulary': vocabulary,
                             'merges': [[lookup[a], lookup[b], lookup[a + b]] for a, b in raw['merges']],
                             'unknown': 0, 'paragraph': 1},
               'stats': {'parameters': report['parameters']},
               'training': report, 'edits': {'version': 4, 'weights': {}}}
    text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    (ROOT / 'candidate.json').write_text(text)
    tokenizer = Tokenizer.from_str(tokenizer_text)
    architecture = report['architecture']
    net = StoryGRU(**architecture)
    net.load_state_dict(state)
    net.eval()
    cases = []
    with torch.no_grad():
        for story in json.loads((ROOT / 'test.json').read_text())[::12]:
            prompt = story['paragraphs'][0]
            ids = [1] + tokenizer.encode(' '.join('▁' + word for word in prompt.split())).ids
            logits, hidden = net(torch.tensor([ids]))
            cases.append({'prompt': prompt, 'ids': ids,
                          'hidden': hidden[0, 0].tolist(), 'logits': logits[0, -1].tolist()})
    (ROOT / 'reference.json').write_text(json.dumps({'sha256': hashlib.sha256(text.encode()).hexdigest(),
                                                   'cases': cases}, separators=(',', ':')))
    print(json.dumps({'checkpointBytes': len(text.encode()), 'parameters': report['parameters']}))


if __name__ == '__main__':
    main()
