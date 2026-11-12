# Pocket LM 2

A self-contained, interactive language model that runs entirely in the browser. Its recurrent neural controller and output weights are hand-written in `index.html`; there is no model download, API, build step, or training step.

**[Open the demo](https://jenschristianschroder.github.io/Local-Model/)**

Open `index.html` locally to use it offline. The **Download HTML** button also saves a standalone copy containing any weight edits made in the playground.

## Explore

- Enter a short prompt or choose an example.
- Generate 32–512 tokens per batch, across sentences and paragraphs. **Continue** appends another batch without resetting the recurrent state or random generator, up to 4,096 generated tokens per document.
- Step through individual tokens, pause generation, or opt into stopping after one sentence.
- Inspect actual sampling probabilities after grammar, repetition, and top-p filters; click a candidate to choose it manually.
- Change temperature, top-p, repetition penalty, theme, and seed to explore sampling. A repeated-trigram guard is enabled by default.
- Edit output weights and observe their effect on probabilities.
- Inspect sentence state, theme memory, active output features, and the complete vocabulary.
- Download the generated text or a standalone HTML copy with your current weight edits.

## What the model is

A word-level **recurrent neural controller with grammar-guided decoding**. It has 462 tokens, 44,268 numeric parameter slots, 42 ReLU transition gates, 23 grammar states, six decaying theme-memory values, and 45 output features. Parameter counts include zero-valued matrix entries.

The grammar is compiled into numeric matrices: ReLU gates match the previous state and token features, then a projection produces the next state. A recurrent theme vector, a fixed prompt-theme anchor, subject features, and object preferences contribute to output logits. An explicit grammar-support mask, simple article constraints, verb–adverb compatibility rules, repetition controls, temperature, and top-p determine the distribution used for sampling. These constraints are visible in the source; this is not an unconstrained trained RNN or transformer.

The decoder remembers the current subject's grammatical number and the current verb. Adverbs must fit that verb: for example, `hums softly`, `blooms slowly`, and `sparkles brightly`. In this limited grammar, `together` requires a plural subject, such as `the robots hum together`. A new clause resets this number check, and objects do not overwrite it. The grammar does not support companion constructions such as `the robot hums together with a friend`, so this is deliberately a narrower rule than English as a whole. These checks happen before sampling and remain active when weights or temperature change, or when the repeated-trigram guard relaxes.

The decoder samples every word individually. It does not select stored replies, assemble sentence templates, or contain a training corpus. Punctuation transitions to a new sentence instead of emitting an end token. State updates are incremental; continuing does not replay the document. Repetition controls retain only the most recent 64 tokens. The repeated-trigram guard relaxes if it would block every otherwise valid candidate.

This remains an educational toy, **not a pretrained LLM or general assistant**. Its limited grammar and vocabulary cannot answer arbitrary questions or ensure semantic coherence. Meaning can drift and sentences can still be odd. It has no learned world knowledge or attention mechanism. Unknown input stays visible in the prompt but does not update the grammar state. A larger parameter count and longer output do not imply pretrained-model quality.

All inference uses plain JavaScript on the CPU. The page makes no inference or asset requests. No dependencies, tracking, or API keys are included.

## Verification

Run `node --test test.mjs` with Node.js 24. The tests execute the actual inline numerical implementation. They cover matrix dimensions, normalized probabilities, 512-token runs over five prompts, greedy and high-temperature runs, continuation equivalence, recurrent memory, grammar disambiguation, subject-number tracking across clauses and modals, verb–adverb compatibility, the four reported singular-subject regressions (including boosted weights), article constraints, top-p, repetition penalties, phrase blocking, deterministic sampling, editable weights, and versioned export overrides.

## Publishing

GitHub Pages uses **GitHub Actions** as its source. `.github/workflows/pages.yml` runs on pushes to `main` and can also be started manually. It runs the numerical checks, copies only `index.html` and `.nojekyll` into the site artifact, then deploys that artifact to the `github-pages` environment.
