"""Train a small conditional language model from random initialization."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "rap-battle"
SPECIAL = ["<pad>", "<unk>", "<start>", "<cue>", "<verse>", "<end>", "<copy>"]
PAD, UNKNOWN, START, CUE, VERSE, END, COPY = SPECIAL
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TOKENS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,!?;:\n-]")
EMBEDDING = 40
HIDDEN = 72
PERFORMERS = {
    "laptop": ("corpus.json", 2041),
    "cloud": ("corpus.json", 2047),
    "robot": ("robot-toaster.json", 2053),
    "toaster": ("robot-toaster.json", 2063),
    "pigeon": ("pigeon-variety.json", 2141),
    "raccoon": ("expanded-roster.json", 2083),
    "librarian": ("expanded-roster.json", 2087),
    "haunted-printer": ("expanded-roster.json", 2089),
    "disco-ball": ("expanded-roster.json", 2099),
    "grandfather-clock": ("expanded-roster.json", 2111),
    "cactus": ("expanded-roster.json", 2113),
    "octopus": ("expanded-roster.json", 2129),
    "weather-forecaster": ("expanded-roster.json", 2131),
    "volcano": ("expanded-roster.json", 2137),
    "challenge": ("challenges.json", 2153),
}


def tokens(text):
    return TOKENS.findall(text.lower())


def examples(corpus, performer):
    opponents = [role for role in corpus["openers"] if role != performer]
    reply_tokens = [[*tokens(before), COPY, *tokens(after), END]
                    for before, after in (reply.split("{cue}") for reply in corpus["replies"][performer])]
    train = [([START, VERSE], tokens(verse) + [END]) for verse in corpus["openers"][performer]]
    valid = [([START, VERSE], tokens(verse) + [END])
             for verse in corpus["validationOpeners"][performer]]
    vocabulary = sorted({word for role in corpus["openers"]
                         for verse in corpus["openers"][role] + corpus["replies"][role]
                         for word in WORDS.findall(verse.lower())} - {"cue"})
    rng = random.Random(2043)
    variants = 24 if len(opponents) == 1 else 3
    past_verses = [
        verse for opponent in opponents for verse in corpus["openers"][opponent]
    ] + [
        reply.format(cue=cue) for opponent in opponents for reply in corpus["replies"][opponent]
        for cue in rng.sample(vocabulary, variants)
    ]
    held_out = [verse for opponent in opponents for verse in corpus["validationOpeners"][opponent]]
    common = set("the and your you that when with from this have their then were them like there what they will just into back while".split())
    for split, verses in ((train, past_verses), (valid, held_out)):
        for verse in verses:
            cues = sorted(set(WORDS.findall(verse.lower())) - common)
            for cue in cues:
                for number, reply in enumerate(reply_tokens):
                    # Keep some cue and reply combinations out of the training split.
                    digest = hashlib.sha256(f"{cue}:{number}".encode()).digest()
                    if (digest[0] % 5 == 0) != (split is valid):
                        continue
                    if split is train and rng.random() > 0.19:
                        continue
                    prompt = [START, *tokens(verse), CUE, cue, VERSE]
                    split.append((prompt, reply))
    return train, valid


def challenge_examples(corpus, split):
    return [([START, f"<{role}>", VERSE], tokens(verse) + [END])
            for role, verses in corpus.items() for verse in verses[split]]


class RapGRU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = nn.Embedding(size, EMBEDDING)
        self.gru = nn.GRU(EMBEDDING, HIDDEN, batch_first=True)
        self.output = nn.Linear(HIDDEN, size)

    def forward(self, ids):
        hidden, _ = self.gru(self.embedding(ids))
        return self.output(hidden)


def batch(examples, lookup):
    width = max(len(prompt) + len(reply) - 1 for prompt, reply in examples)
    inputs = torch.full((len(examples), width), lookup[PAD], dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for row, (prompt, reply) in enumerate(examples):
        ids = [lookup.get(token, lookup[UNKNOWN]) for token in prompt + reply]
        inputs[row, :len(ids) - 1] = torch.tensor(ids[:-1])
        # Train continuation only. The supplied opponent verse is context.
        targets[row, len(prompt) - 1:len(ids) - 1] = torch.tensor(ids[len(prompt):])
    return inputs, targets


def loss(model, inputs, targets, reduction="mean"):
    return F.cross_entropy(model(inputs).transpose(1, 2), targets, reduction=reduction)


@torch.inference_mode()
def evaluate(model, data, lookup):
    total, count = 0.0, 0
    for start in range(0, len(data), 48):
        inputs, targets = batch(data[start:start + 48], lookup)
        total += loss(model, inputs, targets, "sum").item()
        count += (targets != -100).sum().item()
    return total / count


def pack(array):
    if array.ndim == 1:
        return {"encoding": "float32-le", "data": base64.b64encode(array.astype("<f4")).decode()}
    scales = np.maximum(np.max(np.abs(array), axis=1) / 127, 1e-12).astype("<f4")
    values = np.rint(array / scales[:, None]).clip(-127, 127).astype(np.int8)
    return {"encoding": "int8-row", "shape": list(array.shape),
            "data": base64.b64encode(values).decode(), "scales": base64.b64encode(scales).decode()}


def train(performer, steps):
    filename, seed = PERFORMERS[performer]
    raw = (HERE / filename).read_bytes()
    corpus = json.loads(raw)
    if performer == "challenge":
        training = challenge_examples(corpus, "train")
        validation = challenge_examples(corpus, "validation")
        opening_count = 0
    else:
        training, validation = examples(corpus, performer)
        opening_count = len(corpus["openers"][performer])
    counts = Counter(word for prompt, reply in training for word in prompt + reply)
    vocabulary = SPECIAL + sorted(counts.keys() - set(SPECIAL))
    lookup = {word: index for index, word in enumerate(vocabulary)}
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = RapGRU(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    opening, replies = training[:opening_count], training[opening_count:]
    opening_batch = 3 if opening else 0
    history = [{"step": 0, "validationLoss": round(evaluate(model, validation, lookup), 4)}]
    print(performer, history[0], flush=True)
    for step in range(1, steps + 1):
        data = rng.choices(opening, k=opening_batch) + rng.choices(replies, k=32 - opening_batch)
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss(model, *batch(data, lookup))
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == steps:
            result = {"step": step, "validationLoss": round(evaluate(model, validation, lookup), 4)}
            history.append(result)
            print(performer, result, flush=True)

    weights = model.state_dict()
    mapping = {
        "embedding": "embedding.weight", "input": "gru.weight_ih_l0",
        "recurrent": "gru.weight_hh_l0", "inputBias": "gru.bias_ih_l0",
        "recurrentBias": "gru.bias_hh_l0", "output": "output.weight", "outputBias": "output.bias",
    }
    tensors = {name: pack(weights[key].numpy()) for name, key in mapping.items()}
    for name, tensor in tensors.items():
        if tensor["encoding"] == "int8-row":
            values = np.frombuffer(base64.b64decode(tensor["data"]), dtype=np.int8).reshape(tensor["shape"])
            scales = np.frombuffer(base64.b64decode(tensor["scales"]), dtype="<f4")
            weights[mapping[name]] = torch.from_numpy(values * scales[:, None])
    model.load_state_dict(weights)
    checkpoint = {
        "architecture": {"hidden": HIDDEN, "embedding": EMBEDDING, "vocabulary": len(vocabulary)},
        "vocabulary": vocabulary,
        "tensors": tensors,
        "training": {
            "performer": performer, "source": f"training/rap-battle/{filename}", "dataLicense": "CC0-1.0",
            "corpusSha256": hashlib.sha256(raw).hexdigest(), "seed": seed, "steps": steps,
            "parameters": sum(weight.numel() for weight in weights.values()),
            "trainingExamples": len(training), "validationExamples": len(validation),
            "history": history, "quantizedValidationLoss": round(evaluate(model, validation, lookup), 4),
            "torch": torch.__version__,
        },
    }
    (OUTPUT / f"{performer}.json").write_text(json.dumps(checkpoint, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("performer", choices=PERFORMERS)
    parser.add_argument("--steps", type=int, help="Defaults to 600 for the expanded roster, 800 for the original four, and 1000 for the challenge model.")
    args = parser.parse_args()
    expanded = PERFORMERS[args.performer][0] not in {"corpus.json", "robot-toaster.json"}
    torch.set_num_threads(1 if expanded else 2)
    default_steps = 1000 if args.performer == "challenge" else (600 if expanded else 800)
    train(args.performer, args.steps if args.steps is not None else default_steps)
