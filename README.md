# Merry Nonsense

An absurd Christmas story writer powered by a tiny trained language model. Expect talking puddings, confused reindeer, bureaucratic vegetables, and occasional visits from the moon. Open [Merry Nonsense](https://jenschristianschroder.github.io/Local-Model/) or download `index.html` and open it locally. No server, API key, external JavaScript, or model download is needed after loading.

This Christmas edition fine-tunes the trained Pocket LM GRU on 105 original absurd Christmas stories, with 12 separate stories for validation. The Christmas style is learned in its weights. There is no hidden prompt, Christmas keyword boost, story lookup, or output replacement. The four visible starters are ordinary editable user prompts. The inference code contains matrix multiplication, GRU gates, BPE tokenization, softmax, and seeded sampling. There are no grammatical masks, repetition penalties, templates, candidate rejection, or sentence repair. All 1,924 output IDs, including special tokens, participate in softmax.

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

Christmas adaptation uses whole stories, batch size 24, a maximum sequence of 160 tokens, and 80% Christmas batches mixed with 20% base-story batches. Training tracks a validation score using 85% Christmas paragraph loss and 15% base-story loss. The released writer uses the final 1,400-step adaptation checkpoint after qualitative sample review: its openings showed stronger genre adoption than the earlier validation-best checkpoint. Its held-out loss is worse than that earlier checkpoint, and it can reproduce or recombine training passages. This is a small, deliberately narrow model, not evidence of broad generalization. The selected checkpoint and actual loss measurements are in `training/report.json`; the page displays a validation-loss chart. Validation loss measures prediction of held-out tokens, not the truth or coherence of generated passages. Both the descriptive and Christmas validation samples are small and related to their fine-tuning topics. It is not a broad language benchmark. Fine-tuning can improve those topics while slightly worsening broader story prediction.

**This remains a small experimental completion model.** Its training stories contain deliberate absurdity and jokes, but the model is not a reliable joke writer. It can write simple sentences and longer passages, but it can repeat, change subjects, lose character identity, invent words, and make grammatical or factual mistakes. It has no instruction tuning and is not a general chat assistant. Grammar and continuity errors remain limitations of the model, even in a surreal story. A 4,096-token output limit does not imply that it retains 4,096 tokens of reliable context: the GRU compresses history into 192 activations.

## Inspect and edit

- Choose a Christmas starter or write your own opening, then choose a temperature, seed, and batch length. Temperature zero picks the most likely output.
- Continue preserves the hidden state and random generator. One-sentence mode only pauses after generated punctuation; it does not change probabilities or repair the text.
- Probability buttons let the user explicitly choose a token. Automatic generation samples directly from the model.
- Inspect any of the 192 hidden neurons and any output connection. The diagram displays actual activations and contributions.
- Change a weight, restore the original trained values, or download a standalone HTML page or checkpoint with the edits.
- “Recent tokens” displays the last eight tokens; it is not the model's context window.

The matrices use per-row signed int8 quantization with float32 scales; biases remain float32. The browser expands them into float32 arrays. Export evaluates the dequantized tensors, so the measured loss describes the network that actually ships. Casing and punctuation are learned; decoding simply joins pieces, restores whitespace markers, and renders paragraph tokens.

## Verify and deploy

```sh
node build-model.mjs --check
node --test test.mjs
```

`build-model.mjs` only verifies the embedded trained checkpoint. It does not generate weights or compile language rules. Tests compare the exact browser tokenizer, recurrent states, and all logits with independent PyTorch reference vectors. They also verify unfiltered softmax, weight editing, seeded continuation, long-generation stability, checksums, and the whole-page size budget.

GitHub Pages uses the existing **Actions deployment** in `.github/workflows/pages.yml`. A push to `main` runs verification and tests, then deploys only `index.html` and `.nojekyll`. Training data, scripts, and test fixtures are not deployed.

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

The original, AI-authored Christmas corpus is in `training/christmas.txt`, with held-out stories in `training/christmas-valid.txt`. They are only used offline during training and are never embedded as runtime stories. `training/adapt.py` performs the adaptation; `training/christmas-run.json` records the complete adaptation run, including checkpoints after the selected one; `training/report.json` records the corpus checksums, learned merges, loss history, and selected step.
