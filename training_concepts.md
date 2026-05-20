# OmniVoice Training Terms Explainer

This note explains how to understand our OmniVoice Filipino PLD fine-tuning setup, especially if you are used to image segmentation training where the setup is usually `batch_size` plus `epochs`.

## Why OmniVoice Feels Different

Image segmentation training usually looks like this:

```text
dataset size = N images
batch size = B images
1 epoch = N / B batches
train for 100 epochs
```

For example:

```text
10,000 images / batch_size 8 = 1,250 batches per epoch
100 epochs = 125,000 optimizer steps
```

OmniVoice is closer to language-model or speech-token training. Audio samples have very different lengths. One utterance may be 1 second, another may be 10 seconds. After tokenization, these become sequences with very different token counts.

Because of this, OmniVoice does not mainly control memory with a simple fixed sample batch size. It uses token-based batching.

## Sample

A sample is one training item:

```text
one WAV file + one transcript + language_id
```

For our full Filipino PLD split:

```text
train: 33,448 samples
dev:    4,506 samples
test:   4,322 samples
```

## Audio Tokens

OmniVoice does not train directly on raw `.wav` files. Before training, the audio is converted into discrete audio tokens using:

```python
AUDIO_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"
```

The rough pipeline is:

```text
raw WAV + transcript
        v
audio tokenizer
        v
discrete audio tokens + text metadata
        v
WebDataset shards
        v
OmniVoice training
```

This is why we split tokenization into a separate reusable Kaggle Dataset. Tokenization can take around 50 minutes for the full split, but the resulting token shards can be reused across training runs.

## Microbatch

A microbatch is one actual forward/backward pass on the GPU.

In image segmentation, this is similar to:

```text
one batch of images sent through the model
```

In OmniVoice, a microbatch is controlled by token budget:

```python
batch_tokens = 1024
max_batch_size = 4
max_sample_tokens = 1200
```

Meaning:

```text
batch_tokens = 1024
```

Try to keep each microbatch around this token budget.

```text
max_batch_size = 4
```

Never put more than 4 samples in one microbatch.

```text
max_sample_tokens = 1200
```

Drop or skip samples that are too long for this SDPA setup.

The exact number of samples per microbatch can vary. A batch might contain several short utterances or fewer longer utterances.

This is the part that most directly affects GPU memory.

## Gradient Accumulation

Our current full profile uses:

```python
gradient_accumulation_steps = 8
```

This means:

```text
run 8 microbatches
accumulate gradients
then update model weights once
```

So:

```text
8 microbatches = 1 optimizer step
```

Gradient accumulation lets us imitate a larger effective batch without loading that larger batch into GPU memory all at once.

Important: gradient accumulation does not reduce the memory needed by one microbatch. That is why lowering `batch_tokens` and `max_batch_size` fixed the T4 CUDA OOM.

## Step

In OmniVoice, `steps` means optimizer update steps.

Our current full profile uses:

```python
steps = 1000
```

That means:

```text
1000 model updates
```

Because we use:

```python
gradient_accumulation_steps = 8
```

the GPU will run roughly:

```text
1000 optimizer steps x 8 microbatches = 8000 microbatch passes
```

This is different from image segmentation scripts where you may set `epochs = 100` and let the framework calculate steps from dataset size and batch size.

## Epoch

An epoch means one full pass over the training dataset.

In image segmentation:

```text
1 epoch = all images seen once
```

Same concept here, but the number of steps per epoch is less obvious because OmniVoice batches by token count, not by a fixed number of files.

From our earlier T4 subset run:

```text
2,000 train samples ~= 64 optimizer steps per epoch
```

The full training set has:

```text
33,448 train samples
```

Estimated full steps per epoch:

```text
64 x (33,448 / 2,000) ~= 1,070 optimizer steps per epoch
```

So our current full profile:

```python
steps = 1000
```

is approximately:

```text
1000 / 1070 ~= 0.93 epoch
```

In plain terms: the current full run is roughly one pass over the full training set.

## Current Full Profile

Current default in `omnivoice_kaggle_initial.py`:

```python
RUN_PROFILE = "full"
```

Full profile:

```python
"train_samples": None,
"dev_samples": None,
"steps": 1000,
"learning_rate": 2e-5,
"batch_tokens": 1024,
"gradient_accumulation_steps": 8,
"max_batch_size": 4,
"max_sample_tokens": 1200,
"eval_steps": 250,
"save_steps": 1000,
"keep_last_n_checkpoints": 1,
```

Meaning:

```text
train_samples = None
```

Use all 33,448 train samples.

```text
dev_samples = None
```

Use all 4,506 dev samples.

```text
steps = 1000
```

Train for about one full epoch.

```text
batch_tokens = 1024
max_batch_size = 4
```

Small T4-safe microbatches.

```text
gradient_accumulation_steps = 8
```

Accumulate 8 microbatches before each model update.

```text
eval_steps = 250
```

Run dev loss evaluation every 250 optimizer steps.

```text
save_steps = 1000
```

Save only at the final step to avoid Kaggle disk overflow.

## Expected Flow of Current Run

The current full run should behave like:

```text
Train steps 1-250
Evaluate on dev
Train steps 251-500
Evaluate on dev
Train steps 501-750
Evaluate on dev
Train steps 751-1000
Evaluate on dev
Save final checkpoint
Run base/fine-tuned inference samples
Log generated audio and metrics to WandB
```

## Segmentation Analogy

If you are used to image segmentation:

| Image segmentation term | OmniVoice equivalent |
|---|---|
| image | audio/text sample |
| batch size | `batch_tokens` plus `max_batch_size` |
| batch | microbatch |
| gradient accumulation | same concept |
| optimizer step | model update |
| epoch | one pass through all tokenized samples |
| validation every N epochs | `eval_steps` |
| checkpoint every N epochs | `save_steps` |

The biggest conceptual shift:

```text
Segmentation: train for epochs.
OmniVoice: train for optimizer steps.
```

## Approximate Epoch Mapping

For our full split and current batching, estimate:

```text
~1 epoch  ~= 1,000 steps
~2 epochs ~= 2,000 steps
~5 epochs ~= 5,000 steps
~10 epochs ~= 10,000 steps
```

These are approximations. The exact number can shift because token lengths vary and batching is dynamic.

## Runtime Estimate

Earlier T4 speed was roughly:

```text
0.46 to 0.50 optimizer steps/sec
```

For 1000 steps:

```text
1000 / 0.48 ~= 2083 seconds ~= 35 minutes
```

With dev evaluation, checkpoint saving, inference, and WandB upload, a practical estimate is:

```text
45-70 minutes
```

This excludes tokenization. Tokenization is now intended to happen in a separate notebook and be reused as a Kaggle Dataset.

## How To Scale Later

If the current full run fits:

```python
steps = 2000
```

This is roughly 2 epochs.

Then:

```python
steps = 5000
```

This is roughly 5 epochs.

If CUDA memory OOM happens:

```python
batch_tokens = 768
```

or:

```python
batch_tokens = 512
```

Keep:

```python
max_batch_size = 4
```

If disk fills:

```python
save_steps = steps
keep_last_n_checkpoints = 1
SAVE_ACCELERATOR_STATE = False
```

This means save only the final model checkpoint and skip full optimizer-state checkpoints.


