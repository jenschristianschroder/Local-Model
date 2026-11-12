# Pocket LM

A trained language model running entirely in one static HTML file. Open [the live demo](https://jenschristianschroder.github.io/Local-Model/) or download `index.html` and open it locally. No server, API key, external JavaScript, or model download is needed after loading.

Version 4 replaces the earlier hand-authored network with a GRU trained from random initialization. Its language behavior is learned from text. The inference code contains matrix multiplication, GRU gates, BPE tokenization, softmax, and seeded sampling. There are no grammatical masks, repetition penalties, templates, candidate rejection, or sentence repair. All 1,900 output IDs, including special tokens, participate in softmax.

## Model and size

| Property | Value |
|---|---:|
| Learned parameters | 636,908 |
| Vocabulary | 1,900 learned BPE pieces |
| Embedding width | 64 |
| Recurrent layer | One GRU, 192 units |
| Packed tensor payload | 665,872 bytes / 0.666 MB |
| Entire deployed page | Less than 1,000,000 bytes / 1 MB |
| Expanded float32 parameter arrays | 2,547,632 bytes, before other browser memory |
| Generation | Up to 1,024 tokens per batch; Continue up to 4,096 |

MB uses decimal units. Packed tensor size excludes JSON/base64, tokenizer, and UI. The complete page includes all of them. `node build-model.mjs --check` prints the exact current file size and checkpoint SHA-256. Edited downloads can exceed the budget if many overrides are added.

Word pieces can combine into words outside the individual vocabulary entries. This is more flexible than a fixed word list, but it can also produce misspellings or invented words. A token is not necessarily a complete word.

## Training and limits

The base data is a pinned subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): 26,550 training stories, 5,982,352 tokens, and a separate 1,259-story validation subset. The dataset is licensed under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). Dataset revision, byte limits, and SHA-256 checksums are recorded in `training/prepare.py` and `training/report.json`. Raw TinyStories data is not shipped in this repository or webpage. Exact normalized story overlap is excluded from training.

An additional 46 original descriptive passages cover machines, nature, animals, and space. They were written for this project and supplied as ordinary training text, not runtime templates or rules. Six separate passages are used for validation. Both text files are included for inspection; neither is used by the browser. This small collection is deliberately narrow and can be memorized by the model.

The tokenizer is trained only on the base training split. All embeddings, recurrent matrices, biases, and output weights are optimized with next-token cross-entropy and AdamW. Training uses dropout, gradient clipping, and a small amount of input corruption. These are generic training procedures. They do not constrain inference.

The selected checkpoint and actual loss measurements are in `training/report.json`; the page displays a validation-loss chart. Validation loss measures prediction of held-out tokens, not the truth or coherence of generated passages. The descriptive validation sample is small and related to the fine-tuning topics. It is not a broad language benchmark. Fine-tuning can improve those topics while slightly worsening broader story prediction.

**This remains a small experimental completion model.** It can write simple sentences and longer passages, but it can repeat, change subjects, lose character identity, invent words, and make grammatical or factual mistakes. It has no instruction tuning and is not a general chat assistant. A 4,096-token output limit does not imply that it retains 4,096 tokens of reliable context: the GRU compresses history into 192 activations.

## Inspect and edit

- Choose a prompt, temperature, seed, and batch length. Temperature zero picks the most likely output.
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

The final descriptive fine-tuning command and export command are recorded in `training/recipe.md`. Seed 42 is used throughout; numerical results can vary with software and CPU kernels. The base run used random contiguous windows and batch size 32. Checkpoints save optimizer and random-generator state; `--reset-torch-rng` reproduces the phase initialization used in the released run. Intermediate floating-point checkpoints are not committed; the complete quantized checkpoint lives in the HTML.

`training/export.py` quantizes the selected tensors, measures loss before and after quantization, exports PyTorch parity fixtures, and embeds the JSON payload. It refuses to write an HTML file over 1 MB. Copy its report and reference JSON into `training/` before running the verifier.
