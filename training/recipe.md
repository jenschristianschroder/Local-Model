# Training recipes

## Original Pocket LM base

Run the two base phases in the repository README, then this mixed-data phase. Steps are absolute optimizer-step numbers. Resume from the 8,000-step base checkpoint, not from another experiment.

```sh
python training/train.py --data training/data --output training/runs/descriptive --resume training/runs/base/latest.pt --steps 9200 --length 96 --threads 2 --lr .0007 --restart-schedule --reset-torch-rng --extra training/data/descriptions-train.npy --extra-valid training/data/descriptions-valid.npy --extra-fraction .3 --evaluate-every 200
python training/export.py --checkpoint training/runs/descriptive/best.pt --data training/data --output training/runs/export --html index.html
cp training/runs/export/training-report.json training/report.json
cp training/runs/export/reference.json training/reference.json
node build-model.mjs --check
node --test test.mjs
```

The selected checkpoint is step **8,400**. The mixed-data run continued to 9,200, but later checkpoints overfit the small descriptive collection. During fine-tuning, 30% of batches come from descriptions and 70% from TinyStories. Selection minimizes `0.7 * story validation loss + 0.3 * descriptive validation loss`, with validation data never used for gradient updates. The story score uses 12 fixed random validation batches; the description score uses 8. Final export checks story loss on 24 fixed batches, so its reported number differs slightly from the selection score. Loss is cross-entropy in natural-log units per BPE token.

| Phase | Steps | Window | Learning-rate schedule |
|---|---:|---:|---|
| Base training from random weights | 1–4,000 | 64 tokens | cosine, base 0.003, floor 20% |
| Continued base training | 4,001–8,000 | 96 tokens | cosine with absolute step / 8,000, base 0.0015 |
| Descriptive fine-tuning | 8,001–9,200 | 96 tokens | restarted cosine, base 0.0007, floor 20% |

Batch size 32; random seed 42; two CPU threads; dropout 0.08; AdamW weight decay 0.01; gradient norm clipped at 1; input corruption probability 0.003. All model tensors are learned. The first two training phases predated saving Torch's RNG state, so the reproduction commands explicitly reset the phase initialization rather than restoring that state.

A separate whole-paragraph fine-tuning experiment was also evaluated, starting again from step 8,000, using length 128, 50% descriptive batches, and the same 0.0007 restarted schedule. It produced some clearer sample openings but worse combined held-out loss. It is not the shipped checkpoint. Neither experiment imposes language rules during generation.

The initial validation loss of 7.5505 uses eight fixed batches. Final quantized loss uses 24 batches. These are evidence that the model learned, not a controlled comparison with the prior hand-wired model. The six descriptive validation documents are too small and narrow to establish general language proficiency. They were authored for the project alongside the training passages but are not included in training.

Raw training data and floating-point optimizer checkpoints stay outside git. The exact released quantized network, tokenizer, training summary, and editable-output overrides are embedded in `index.html`. Its SHA-256 is recorded in the training report and independent PyTorch parity fixture.


## Merry Nonsense Christmas edition

This section reproduces the initial Christmas release. The continuation update below starts from that published model.

The Christmas edition starts with the previous committed quantized checkpoint, learns 24 additional BPE merges from Christmas training prose, and fine-tunes every model parameter. Download the original base HTML and its report from the exact commit below; these are training inputs, not runtime network dependencies.

```sh
curl -L https://raw.githubusercontent.com/jenschristianschroder/Local-Model/0d1d4c00484bdea30877afe699a831fb2bc2352c/index.html -o training/data/base-v4.html
curl -L https://raw.githubusercontent.com/jenschristianschroder/Local-Model/0d1d4c00484bdea30877afe699a831fb2bc2352c/training/report.json -o training/data/base-v4-report.json
python training/prepare.py --data training/data --vocab 1900
python training/adapt.py --base-html training/data/base-v4.html --base-report training/data/base-v4-report.json --data training/data --output training/runs/christmas --steps 1400 --length 160 --batch 24 --threads 2 --seed 1225 --lr .001 --fraction .8 --merges 24
python training/export.py --checkpoint training/runs/christmas/latest.pt --selection latest --data training/data --output training/runs/export --html index.html
cp training/runs/export/training-report.json training/report.json
cp training/runs/export/reference.json training/reference.json
node build-model.mjs --check
node --test test.mjs
```

Start with the original tokenizer produced by `prepare.py` for each fresh adaptation run; `adapt.py` extends it in place and retokenizes the pinned base corpus. It validates the original vocabulary against the base checkpoint before doing so. New rows use the mean of their two constituent rows as initialization; all are then learned by gradient descent. Corpus and model dimensions are recorded in the report.

Original AI-authored training text: 105 separate Christmas stories. Validation: 12 separate Christmas stories. The vocabulary merges are learned only from the training text. AdamW, dropout 0.08, gradient clipping at 1, input corruption 0.003, and cosine learning-rate decay to 15% are generic offline training procedures. No training procedure adds a runtime language constraint.

Each Christmas batch contains complete stories padded to a fixed length; padding targets are ignored, and a paragraph marker ends each story. 80% of batches use Christmas text, with 20% base stories to retain general language patterns. Training records the checkpoint minimizing `0.85 * Christmas full-paragraph loss + 0.15 * base-story validation loss`. The released writer instead uses the final checkpoint after qualitative review of the four visible starters. Its more recognisable Christmas openings come with a higher held-out loss and more memorization. Both results are disclosed in `releaseSelection` and the full run record. Final export evaluates both float and quantized Christmas loss. A small validation set measures prediction of related prose, not general coherence or how funny the output is.

An initial adaptation without vocabulary extension was also tested. It improved Christmas token prediction, but produced fragmented festive words in samples. The released run includes the learned vocabulary extension. The report's `holiday.selectedAdaptationStep` is 1,400. `holiday.bestAdaptationStep` identifies the earlier validation-best checkpoint. `training/christmas-run.json` preserves the complete run, including the evidence of overfitting.

## Paragraph continuation and repetition update

Use the prepared **1,924-piece** tokenizer and retokenized data from the Christmas recipe above. The old short-story examples only supervised a paragraph marker at their ends; they did not train the Christmas model to continue after it. The update packs stories into continuous 320-token training rows and includes 20 new three-paragraph stories. Four separate longer stories are used only for evaluation. Literal story and paragraph separators are training-data structure, not runtime templates.

The training objective combines ordinary next-token cross-entropy with sequence unlikelihood, inspired by [Welleck et al.](https://arxiv.org/abs/1908.04319). Every fourth optimizer step generates four continuations from prefixes drawn from the training texts. Tokens that complete an already-seen 16-token span receive an offline negative training loss. No gradient is taken through the discrete generation choices. Cross-entropy and unlikelihood both update the same learned tensors. All runtime generation remains unfiltered softmax and seeded sampling.

```sh
curl -L https://raw.githubusercontent.com/jenschristianschroder/Local-Model/dc4d4cadb4939e17e95056c7fd80aa5a60f5de14/index.html -o training/data/before-repetition.html
curl -L https://raw.githubusercontent.com/jenschristianschroder/Local-Model/dc4d4cadb4939e17e95056c7fd80aa5a60f5de14/training/report.json -o training/data/before-repetition-report.json
python training/repair.py --base-html training/data/before-repetition.html --base-report training/data/before-repetition-report.json --data training/data --output training/runs/continuation --steps 450 --lr .0002 --alpha .5 --base-fraction .2 --long-fraction .25 --ngram 16 --rollout 768 --repeat-target end
python training/export.py --checkpoint training/runs/continuation/step-375.pt --data training/data --output training/runs/export --html index.html
cp training/runs/export/training-report.json training/report.json
cp training/runs/export/reference.json training/reference.json
node training/repetition-eval.mjs index.html repetition-results.json
node training/repetition-eval.mjs index.html additional-results.json additional
node build-model.mjs --check
node --test test.mjs
```

Selection uses **375 additional updates**, absolute step 10,175. The run continued to 450, but that later compressed checkpoint looped much more in the long browser-kernel evaluation. The selected model was checked after quantization, since small rounding differences can change greedy paths substantially. The selected run uses float-model rollouts during training; an additional experiment with quantized rollouts is supported by `--quantized-rollouts` and recorded with the other comparisons. Neither numerical development score nor held-out loss alone determined release selection.

Batch size 12; two CPU threads; seed 923; AdamW with weight decay 0.01; dropout 0.08; gradient clipping at 1; input corruption 0.003. Training rows are 20% base-story windows and 80% Christmas sequences. Within Christmas sequences, document selection is 25% longer stories and 75% original short stories. The unlikelihood coefficient is 0.5; sampled negative continuations use temperature zero 80% of the time and 0.55 otherwise. The learning rate follows a 450-step cosine schedule with a 20% floor. Reproducing only 375 schedule steps with a different total changes the result.

`training/repetition-report.json` stores full before/after outputs, metric definitions, the unchanged inference-source checksum, and additional prompt checks. `training/repetition-experiments.json` records compared training runs and checkpoint decisions. The data is small and can be memorized. Lower repetition does not guarantee grammatical sentences, a continuous plot, or novel jokes.
