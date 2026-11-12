# Pocket LM

A self-contained, interactive neural language model that runs entirely in the browser. Every weight is assigned directly in `index.html`; there is no model download, API, build step, or training step.

**[Open the demo](https://jenschristianschroder.github.io/Local-Model/)**

Open `index.html` locally to use it offline. The **Download HTML** button also saves a standalone copy containing any weight edits made in the playground.

## Explore

- Enter a short prompt or choose an example.
- Generate a continuation, or step through it one token at a time.
- Inspect the actual next-token probability distribution; click a candidate to choose it manually.
- Change temperature and seed to explore sampling.
- Edit output weights and observe their effect on probabilities.
- Inspect the active hidden neurons, four-token context, and complete vocabulary.

## What the model is

A word-level feed-forward neural language model, with hand-wired feature embeddings, four tokens of context, 18 ReLU hidden units, and a softmax output. The model constructs explicit numeric matrices and performs matrix-vector operations for each prediction. Its behavior is deliberately encoded in the weights. It does not select canned replies.

It is an educational toy, **not a pretrained LLM or general assistant**. It cannot answer arbitrary questions. Its vocabulary is limited; unknown input maps to `<unk>`, and generated text can be ungrammatical. It has no attention, long-term memory, learned world knowledge, or training corpus. Embeddings and weights are visible in the inline script. Parameter counts include zero-valued matrix entries.

All inference uses plain JavaScript on the CPU. The page makes no inference or asset requests. No dependencies, tracking, or API keys are included.

## Verification

Run `node --test test.mjs` with a recent Node.js release. Tests exercise the actual inline numerical implementation: matrix dimensions, probability normalization, special-token masking, context limits, deterministic sampling, temperature behavior, editable weights, and saved override bounds.

## Publishing

GitHub Pages uses **GitHub Actions** as its source. `.github/workflows/pages.yml` runs on pushes to `main` and can also be started manually. It runs the numerical checks, copies only `index.html` and `.nojekyll` into the site artifact, then deploys that artifact to the `github-pages` environment.
