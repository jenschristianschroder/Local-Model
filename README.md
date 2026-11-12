# Pocket LM 3

A self-contained demonstration of a **hand-wired recurrent neural network** running entirely in the browser. All language behavior is represented by numeric model parameters and hidden activations. The inference engine has no grammar checks or output masks.

**[Open the demo](https://jenschristianschroder.github.io/Local-Model/)**

Open `index.html` locally to use it offline. No dependencies, model downloads, API keys, tracking, or server-side inference are involved.

## What changed

Version 2 applied grammar, article, verb–adverb, and repetition filters outside the network. Version 3 removes that decoding layer. The page embeds a frozen numeric checkpoint; a small generic inference kernel evaluates it without reading word labels, parts of speech, or compatibility tables.

The model now represents syntax, subject number, article preferences, verb affinities, sentence length, themes, and recent-token activity in its recurrent activations. Output weights turn these features into **finite logits for every vocabulary ID**. Sampling uses the full softmax distribution, with temperature as the only distribution control. There is no top-p pruning, phrase blacklist, candidate rejection, or fallback grammar filter.

For example, the connection from the `Single / unknown subject` activation to the `together` output has weight **−24**. That lowers its score in “the robot hums,” while plural subjects leave this activation off. This is a preference in the model, not an absolute prohibition. Click **Inspect “together”**, then change that weight to **+16**: the previously discouraged word becomes the model's top prediction. Similarly, changing syntax or verb-affinity weights can make the model produce incorrect language.

## Numeric architecture

- 462 vocabulary IDs, including input special tokens.
- A sparse 462 × 541 token embedding matrix.
- A recurrent cell with 806 ReLU neurons and 558 hidden activations.
- A sparse linear output layer producing 462 finite logits.
- 10,782 nonzero weights and biases. This excludes structural zeros; v2 displayed dense parameter slots, so the two counts are not directly comparable.

At each step, the network concatenates the previous hidden vector and the current token embedding. Its cell evaluates an ordered, acyclic set of affine/ReLU neurons; numeric readout indices select the next hidden vector. A linear layer produces logits, and softmax converts them into probabilities. The kernel is marked `INFERENCE KERNEL START` / `END` in `index.html` and can run with just the numeric checkpoint, without the DOM or label metadata.

The sparse rows are flat arrays of `[column, weight, column, weight, ...]`. The checkpoint contains embeddings, cell connections, cell biases, readout indices, initial activations, output connections, and output biases. Labels are separate inspection metadata. `build-model.mjs` authors the weights offline and embeds the checkpoint into the HTML; **the browser does not execute this builder**. Its human-authored language knowledge is compiled into numeric connections, not learned from training data.

Repetition feedback is part of the network: decaying token activations feed negative output connections. It is a soft tendency, not a guarantee against repeated phrases. All vocabulary entries receive scores, including special tokens. At very low temperature, floating-point softmax can underflow extremely small probabilities; temperature zero selects the largest logit.

## Explore

- Start with a short prompt or one of the four examples.
- Generate 32–512 tokens per batch. Continue preserves hidden state and the random sequence, up to 4,096 generated tokens per document.
- Step through predictions, inspect raw logits and probabilities, and edit any output connection.
- Try **Inspect “together”** to see the original grammar issue represented by an editable weight.
- Change temperature and seed. Length, Stop, and One sentence only govern when generation pauses, not which word is chosen.
- **Download weights** exports the complete numeric checkpoint plus inspection labels.
- **Download HTML** saves a standalone page including your output-weight edits. Version 3 uses its own override format; version 2 edits have incompatible feature indices and are ignored.

Unknown input maps to the numeric `<unk>` embedding. Display formatting adds spacing, capitalization, and paragraph breaks without replacing words. The last eight displayed tokens are UI history and have no effect on inference.

This is an educational model with hand-authored weights, **not a pretrained LLM or general assistant**. Its representations and vocabulary are limited. It can still produce awkward language, drift between themes, repeat itself, and make grammar mistakes. Longer output and more internal activations do not give it general world knowledge.

## Reproduce and verify

With Node.js 24:

```sh
node build-model.mjs
node --test test.mjs
```

Use `node build-model.mjs --check` to confirm the embedded checkpoint matches the authoring script without rewriting the HTML.

The 23 tests execute the actual inline kernel and frozen checkpoint. They verify raw-logit/softmax equality, numeric-only execution without labels, uniform predictions when output weights are zeroed, reversal of grammar preferences by editing weights, embedding swaps, subject memory, verb affinities, finite article preferences, recurrent repetition feedback, 512-token generation, continuation, seeded sampling, and checkpoint/HTML edit round trips. These tests establish that grammar behavior comes from the numeric model rather than an external filter.

## Publishing

GitHub Pages uses **GitHub Actions**. `.github/workflows/pages.yml` verifies the embedded checkpoint and runs the tests on pushes to `main`, then deploys only `index.html` and `.nojekyll`. The weight-authoring script and test files are not part of the deployed site artifact.
