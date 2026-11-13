# Synthetic coherence experiment

This candidate trains a fresh GRU and a fresh BPE tokenizer using text generated
for this experiment. The corpus generator has no file inputs or network calls.
It contains six newly AI-authored four-paragraph Christmas plots, expanded by
character, companion, and object substitutions. Existing checkpoints, tokenizers,
stories, and external datasets are excluded from candidate training.

There are 864 training stories, 144 validation stories, and 144 test stories.
Actor/companion pairs are disjoint between splits. Plot templates are shared.
This deliberately narrow experiment measures learning continuity and recombining
known plots. It cannot establish general storytelling quality.

The model uses 256 GRU units and 64 embedding dimensions. Training processes
complete stories, preserving the recurrent state and gradients across paragraph
boundaries. All parameters start randomly. The 1,024-piece BPE tokenizer learns
only from the training split. Validation loss selects the checkpoint.

The second run assigns 12 times the training loss weight to tokenizer pieces
that occur in character, companion, and object names. This changes gradient
updates during offline training. `initial-training-report.json` and
`initial-evaluation.json` record the first, unweighted 1,000-step run, which
switched character names and objects despite low validation loss.

## Observed result

The weighted run completed 2,000 steps and selected step 1,900 by held-out loss.
The candidate contains 576,000 learned parameters. Its JSON checkpoint is
821,090 bytes. Browser/PyTorch maximum error is below 0.000004.

Across 24 familiar-plot continuations, the main character appears in all three
following paragraphs in 14 cases, and the original object appears in all three
in eight cases. The initial run scored zero for both diagnostics. These counts
do not exclude other characters or establish causal coherence.

Manual review finds companion substitutions, reused plot passages, and drift
after the story ends. All four new plot openings drift into training plots in
greedy generation. This candidate is an unsuccessful release candidate for
general Merry Nonsense use. The production page has not been updated.

## Reproduce

Install CPU PyTorch, NumPy, and tokenizers. The recorded run uses PyTorch
2.15.0+cpu, NumPy 2.5.4, and tokenizers 0.23.3.

```sh
python training/coherence/corpus.py
python training/coherence/train.py --steps 2000
python training/coherence/export_candidate.py
node training/coherence/evaluate.mjs
```

To reproduce the first run, use `--steps 1000 --entity-weight 1` and then export
and evaluate. These commands overwrite the candidate and current reports, so
copy their outputs before reproducing another run.

The trained floating-point checkpoint is written under `training/runs/`.
`candidate.json` contains quantized learned weights and the new tokenizer.
`reference.json` contains independent PyTorch reference outputs for quantized
weights. The evaluator checks these against the exact existing browser kernel.

`evaluation.json` records 512-token continuations for both the released model
and the candidate at temperatures 0 and 0.55. It includes twelve held-out
combinations and four new plots. Entity mentions are diagnostic measurements.
Read the outputs to assess plot continuity, completion, and repetition.

The existing model is loaded only as an evaluation baseline. The candidate
receives no baseline outputs during training. Both models use the existing
browser tokenizer, recurrent computation, softmax, and sampler implementations
with their own learned parameters. Corpus templates run only offline.
