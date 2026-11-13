# Rap Battle training data

The examples in `corpus.json` (Laptop and Cloud), `robot-toaster.json` (Robot and Toaster), and `expanded-roster.json` (the ten additional opponents) were authored for this demo. The examples and their synthetic combinations are dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). No external lyrics, other datasets, or pretrained checkpoints enter the training run.

The original four models were trained in pairs. Each of the ten new models was trained from random initialization using only `expanded-roster.json`. Each new model learns its own replies in the context of verses from the other nine new opponents. You can mix all 14 opponents in the browser. The original models may still reference their training partners.

The trainer pairs a previous verse with one of its words and a two-line response written for the other performer. It also trains opening verses. These combinations are generated offline. The response target uses a dedicated copy token where the callback word belongs. At inference, that token renders as the selected word. The model learns when to emit it and samples the surrounding verse from its weights.

Validation uses separate opponent verses and response/callback pairings excluded from training. This tests transfer to new contexts and recombinations of familiar writing patterns. The corpus is deliberately small, and a model can reproduce training lines or miss the selected callback.
