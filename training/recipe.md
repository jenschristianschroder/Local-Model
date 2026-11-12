# Released training recipe

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
