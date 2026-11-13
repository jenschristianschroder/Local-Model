# Merry Nonsense

An absurd Christmas story writer powered by a tiny trained language model. Expect talking puddings, confused reindeer, bureaucratic vegetables, and occasional visits from the moon. Open [Merry Nonsense](https://jenschristianschroder.github.io/Pocket-LM/) or download `index.html` and open it locally. No server, API key, external JavaScript, or model download is needed after loading.

This Christmas edition fine-tunes the trained Pocket LM GRU on 105 original absurd Christmas stories, with 12 separate stories for validation. The Christmas style is learned in its weights. There is no hidden prompt, Christmas keyword boost, story lookup, or output replacement. The four visible starters are ordinary editable user prompts. The inference code contains matrix multiplication, GRU gates, BPE tokenization, softmax, and seeded sampling. There are no grammatical masks, repetition penalties, templates, candidate rejection, or sentence repair. All 1,924 output IDs, including special tokens, participate in softmax.

The continuation update adds 20 original three-paragraph stories, with four separate longer stories for validation. It also uses **sequence unlikelihood during offline training**: when the model generates a repeated span, a training loss lowers the probability of that span by updating the weights. No repeat detector, penalty, rejection, or text repair runs in the browser. The inference kernel and seeded sampler are unchanged. The training code and both longer-story files are in `training/`.

## Model and size

| Property | Value |
|---|---:|
| Learned parameters | 643,076 |
| Vocabulary | 1,924 learned BPE pieces |
| Embedding width | 64 |
| Recurrent layer | One GRU, 192 units |
| Packed tensor payload | 672,304 bytes / 0.672 MB |
| Entire deployed page | Less than 1,000,000 bytes / 1 MB |
| Expanded float32 parameter arrays | 2,572,304 bytes, before other browser memory |
| Generation | Up to 1,024 tokens per batch; Continue up to 4,096 |

MB uses decimal units. Packed tensor size excludes JSON/base64, tokenizer, and UI. The complete page includes all of them. `node build-model.mjs --check` prints the exact current file size and checkpoint SHA-256. Edited downloads can exceed the budget if many overrides are added.

Word pieces can combine into words outside the individual vocabulary entries. This is more flexible than a fixed word list, but it can also produce misspellings or invented words. A token is not necessarily a complete word.

## Training and limits

The base data is a pinned subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): 26,550 training stories, 5,982,352 tokens, and a separate 1,259-story validation subset. The dataset is licensed under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). Dataset revision, byte limits, and SHA-256 checksums are recorded in `training/prepare.py` and `training/report.json`. Raw TinyStories data is not shipped in this repository or webpage. Exact normalized story overlap is excluded from training.

Before Christmas adaptation, an additional 46 original descriptive passages cover machines, nature, animals, and space. They were generated for this project and supplied as ordinary training text, not runtime templates or rules. Six separate passages are used for validation. Both text files are included for inspection; neither is used by the browser. This small collection is deliberately narrow and can be memorized by the model.

The original 1,900-piece tokenizer was learned from the base training split. This edition adds 24 BPE merges learned from Christmas training text only, making common festive words more compact. New embedding/output rows are initialized from their constituent rows, then optimized along with every other parameter. Validation text does not determine merges. All embeddings, recurrent matrices, biases, and output weights are optimized with next-token cross-entropy and AdamW. Training uses dropout, gradient clipping, and a small amount of input corruption. These are generic training procedures. They do not constrain inference.

The initial Christmas adaptation used whole stories, batch size 24, a maximum sequence of 160 tokens, and 80% Christmas batches mixed with 20% base-story batches. It selected the final 1,400-step checkpoint after qualitative sample review, although an earlier checkpoint had better held-out loss. Its short examples ended with a paragraph marker, with no Christmas supervision after that marker. Long generation could revert to familiar base-story openings and enter a repeating cycle.

The continuation update trains across paragraph and story boundaries, preserves state across those boundaries, and includes negative examples of model-generated repetition. It starts from the previous published quantized weights. Checkpoint selection considers long greedy rollouts, held-out losses, and sampled prose; reducing repetition alone is not enough. Exact settings, corpus checksums, and selected steps are in `training/report.json`. `training/repetition-report.json` records before/after outputs from the exact browser kernel, including remaining failures. These are small regression sets, not evidence of broad generalization or reliable long-story coherence. Validation loss measures held-out token prediction, not grammar, truth, continuity, or good jokes. The model can still reproduce and recombine its training passages.

**This remains a small experimental completion model.** Its training stories contain deliberate absurdity and jokes, but the model is not a reliable joke writer. It can write simple sentences and longer passages, but it can repeat, change subjects, lose character identity, invent words, and make grammatical or factual mistakes. It has no instruction tuning and is not a general chat assistant. Grammar and continuity errors remain limitations of the model, even in a surreal story. A 4,096-token output limit does not imply that it retains 4,096 tokens of reliable context: the GRU compresses history into 192 activations.

## Explore and edit

Pocket LM is a phone-first, eight-chapter learning presentation. Use Back/Next or Chapters to move between lessons. Vertical scrolling always reads the current lesson; an optional horizontal swipe changes chapters outside controls. Browser Back works for chapters and dismisses open details. Reduced motion is respected.

- Start with an opening line and two suggestions. Generation settings and more starters open in labelled detail panels.
- Read the story, continue it, or open More to save it and start again. Entering a lesson or opening details pauses generation without discarding its state.
- Follow a prediction with the sentence ending and three candidates together. Generate one token samples normally; choosing a candidate explicitly overrides that one sample. More predictions shows twelve candidates; every token remains eligible during automatic generation.
- Inspect a portrait memory overview, follow one real connection, or explore all 192 neurons and search the output vocabulary.
- Edit a relevant output connection and compare the current probability with the trained weights at the same model state and temperature. Reset one connection or restore all weights. Exports include edits.
- Training data, loss curves, architecture, and searchable vocabulary are available in optional panels. The essential lessons and model limitations remain visible.
- Prompt, generation state, random-generator position, edits, settings, and chapter are saved to sessionStorage when available. No data leaves the browser. Storage denial does not prevent generation.
- Download app is available in the final chapter on phones and desktop. The standalone HTML runs offline; no development packages are shipped.

Temperature zero chooses the most likely token and ignores the seed. At a nonzero temperature, matching prompt, seed, weights, and settings reproduce a story. Changing the seed resets the continuation. One-sentence mode pauses at generated punctuation without repairing output. Continue preserves the recurrent state and random generator.

The matrices use per-row signed int8 quantization with float32 scales; biases remain float32. The browser expands them into float32 arrays. The most recent eight tokens are a display aid, not a context window. Casing and punctuation are learned; decoding joins pieces and restores whitespace and paragraphs.

### UI source

Edit `ui/body.html`, `ui/style.css`, and `ui/app.js`, then run `npm ci --ignore-scripts` and `node build-ui.mjs`. The build preserves the embedded checkpoint and the numeric kernel byte-for-byte, minifies only the UI, and rejects pages at or above 1 MB. The UI selector and CSS variable aliases in the build keep readable source within the standalone size budget. Edit `ui/examples.json` to update the three introductory examples. Stories are shown in full, with transparent side arrows and small position dots; buttons, arrow keys, and horizontal swipes change stories without advancing chapters. There is no automatic rotation.

`ui/review.html` is a development review harness for 320×568, 390×844, 430×932, and desktop viewports. The deployment exposes it as `review.html` for responsive inspection. It is separate from the standalone app.

## Verify and deploy

```sh
npm ci --ignore-scripts
node build-ui.mjs
node build-model.mjs --check
node --test test.mjs ui.test.mjs
```

`build-model.mjs` only verifies the embedded trained checkpoint. It does not generate weights or compile language rules. Tests compare the exact browser tokenizer, recurrent states, and all logits with independent PyTorch reference vectors. They also verify unfiltered softmax, weight editing, seeded continuation, long-generation stability, checksums, and the whole-page size budget.

Run `node training/repetition-eval.mjs index.html repetition-results.json` for the full 1,024-token repetition regression suite. This evaluator runs offline and is never imported by the webpage. Its repeated-eight-word score counts the fraction of eight-word windows that already appeared earlier in the same output. Exact repeated paragraphs count only paragraphs of at least 20 words. Lower repetition does not necessarily mean better prose.

The released update reduces mean repeated-eight-word spans from **75.0% to 4.7%** across the six greedy regression continuations. Cases with a repeated full paragraph fall from **6/6 to 0/6**. At the default temperature of 0.55, the same measure falls from **3.0% to 0.6%** across 12 seeded continuations. Four additional prompts, tested at three temperatures with two seeds for nonzero temperatures, produced no repeated full paragraphs in 20 further outputs. All runs used 1,024 generated tokens and the compressed model. These results do not guarantee that other prompts or longer continuations cannot loop; grammar and story continuity remain limited.

GitHub Pages uses the existing **Actions deployment** in `.github/workflows/pages.yml`. A push to `main` runs verification and tests, then deploys `index.html`, `.nojekyll`, and the separate responsive review harness. The full training corpus, scripts, and test fixtures are not deployed. The three curated display examples are included in the standalone app.

## Reproduce training

Training is offline; Python and PyTorch are not required to view the page. Python 3.11, NumPy 2.3.5, tokenizers 0.21.2, and PyTorch 2.7.1 CPU were used. Install CPU PyTorch using its official wheel index, then the small dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.3.5 tokenizers==0.21.2
python training/prepare.py --data training/data --vocab 1900
python training/train.py --data training/data --output training/runs/base --steps 4000 --length 64 --threads 2 --lr .003
python training/train.py --data training/data --output training/runs/base --resume training/runs/base/latest.pt --steps 8000 --length 96 --threads 2 --lr .0015 --reset-torch-rng
```

The original descriptive recipe and the released Christmas adaptation command are recorded in `training/recipe.md`. Christmas adaptation can start directly from the previous committed HTML checkpoint, so it does not require repeating the base training. Base training uses seed 42; Christmas adaptation uses seed 1225; numerical results can vary with software and CPU kernels. The base run used random contiguous windows and batch size 32. Checkpoints save optimizer and random-generator state; `--reset-torch-rng` reproduces the phase initialization used in the released run. Intermediate floating-point checkpoints are not committed; the complete quantized checkpoint lives in the HTML.

`training/export.py` quantizes the selected tensors, measures loss before and after quantization, exports PyTorch parity fixtures, and embeds the JSON payload. It refuses to write an HTML file over 1 MB. Copy its report and reference JSON into `training/` before running the verifier.

The original, AI-authored Christmas corpus is in `training/christmas.txt`, with held-out stories in `training/christmas-valid.txt`. They are used offline during training. Three user-selected examples are displayed in the introduction carousel; these also appear in the training corpus and are not evidence of novel output. The carousel text is never fed to inference or used to replace generated text. `training/adapt.py` performs the adaptation; `training/christmas-run.json` records the complete adaptation run, including checkpoints after the selected one; `training/report.json` records the corpus checksums, learned merges, loss history, and selected step.

## Rap Battle

Open [Rap Battle](https://jenschristianschroder.github.io/Pocket-LM/rap-battle/) to choose two opponents from a roster of 14. Play three rounds, pick a callback word after each verse, then vote. The two chosen GRUs run on your device after their checkpoints load. They were trained from random initialization on original CC0 examples created for this demo. The tiny corpus can lead to repeated lines or missed callbacks. See [training data and split details](training/rap-battle/DATA.md).

To run locally, start `python -m http.server 8000` from the repo root and open `http://localhost:8000/rap-battle/`. To reproduce the checkpoints, install CPU PyTorch and NumPy and run:

```sh
python training/rap-battle/train.py laptop
python training/rap-battle/train.py cloud
python training/rap-battle/train.py robot
python training/rap-battle/train.py toaster
python training/rap-battle/train.py haunted-printer
```
