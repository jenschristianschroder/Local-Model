# Rap Battle training data

The examples in `corpus.json` were authored for this demo. Its examples and the synthetic combinations produced from them are dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). No external lyrics, other datasets, or pretrained checkpoints enter the training run.

The trainer pairs a previous verse with one of its words and a two-line response written for the other performer. It also trains opening verses. These combinations are generated offline. The response target uses a dedicated copy token where the callback word belongs. At inference, that token renders as the selected word. The model learns when to emit it and samples the surrounding verse from its weights.

Validation uses separate opponent verses and response/callback pairings excluded from training. This tests transfer to new contexts and recombinations of familiar writing patterns. The corpus is deliberately small, and either model can reproduce training lines or miss the selected callback.
