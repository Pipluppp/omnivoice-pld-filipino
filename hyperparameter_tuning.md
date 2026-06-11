# OmniVoice Filipino Hyperparameter Tuning Notes

These notes record the hyperparameter tuning process for the Filipino PLD fine-tuning runs on Kaggle, from the first exploratory run through the completed controlled learning-rate comparison. The initial 1000-step run showed both `train/loss` and `eval/loss` still trending downward, with eval loss ending near its best value. That did not look like clear overfitting; it more likely meant the run was still short, which motivated the longer 5000-step runs below.

The guidance below is grounded in OmniVoice's own fine-tuning configs and docs, especially:

- `.OmniVoice/examples/config/train_config_finetune.json`
- `.OmniVoice/examples/config/train_config_finetune_sdpa.json`
- `.OmniVoice/docs/training.md`

## Final Controlled Learning-Rate Comparison

The controlled best-eval sweep is complete. All three runs used the identical 5000-step setup and the same checkpoint policy (evaluate the development set at fixed intervals, save only when development loss improves), differing only in learning rate:

| Controlled run | Learning rate | Steps | Checkpoint policy | Training notebook |
| --- | ---: | ---: | --- | --- |
| Best-eval LR 1e-5 | `1e-5` | 5000 | lowest dev loss (step 4900) | `notebooks/omnivoice_training_best_eval.py` |
| Best-eval LR 2e-5 | `2e-5` | 5000 | lowest dev loss | `notebooks/omnivoice_training_best_eval_lr_2e_5.py` |
| Best-eval LR 5e-6 | `5e-6` | 5000 | lowest dev loss | `notebooks/omnivoice_training_best_eval_lr_5e_6.py` |

All four systems were evaluated on the same full 4,322-utterance PLD Filipino test split with the same inference settings, evaluator models, and text normalization. The base and LR 1e-5 rows come from the earlier full evaluation runs; the LR 2e-5 and LR 5e-6 rows come from `notebooks/omnivoice_evaluation_metrics_new_lr_reruns.py`, exported locally to `eval-2e-5-and-5e-6/`.

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint | 21.96 | -0.59 | 0.605 | 3.60 |

Tuning conclusions:

- `1e-5` remains the strongest learning rate overall by WER: 18.52%, a 4.03 absolute-point reduction versus base, while keeping SIM-o slightly above base.
- `2e-5` is very close in WER at 18.83% but trades away speaker similarity: its SIM-o of 0.583 is the only fine-tune value below base.
- `5e-6` is the most conservative update: it has the best SIM-o among the fine-tunes at 0.605 but only a modest 0.59-point WER improvement over base.
- The base model keeps the highest UTMOS at 3.64; the fine-tunes sit close together at 3.60-3.61.

Because the checkpoint policy and step count are now held constant, these differences are attributable mostly to learning rate. The pattern is an intelligibility-versus-speaker-similarity/naturalness tradeoff: a larger learning rate moves the model further toward Filipino intelligibility at some cost to speaker similarity, while a smaller learning rate stays closer to the base model on every axis.

### Project History: Earlier Exploratory Runs

Before the controlled sweep, two exploratory runs used final-step checkpoints instead of development-loss selection: a 1000-step `2e-5` run (`omnivoice-filipino-full-checkpoint-1000`, WER 20.07, SIM-o 0.610, UTMOS 3.60) and a 2000-step `5e-6` run (`omnivoice-filipino-full-checkpoint-2000`, WER 22.64, SIM-o 0.611, UTMOS 3.57). These mixed checkpoint policy, step count, and learning rate, so they were not a clean comparison; they motivated the controlled sweep above and are kept only as history. See `progress/2026-05-19-full-train-track-metrics.md`.

## Current Kaggle Baseline

The current Kaggle notebook-style script uses a T4-friendly SDPA setup:

```python
TRAINING_STEPS = 1000
LEARNING_RATE = 2e-5
BATCH_TOKENS = 1024
GRADIENT_ACCUMULATION_STEPS = 8
MAX_BATCH_SIZE = 4
MAX_SAMPLE_TOKENS = 1200
EVAL_STEPS = 250
SAVE_STEPS = 250
```

The effective token budget per optimizer update is roughly:

```text
BATCH_TOKENS * GRADIENT_ACCUMULATION_STEPS = 1024 * 8 = 8192 tokens
```

This roughly matches the repository fine-tuning config's `batch_tokens = 8192`, but splits it into smaller per-forward batches so it can fit Kaggle's free T4 GPU.

## Highest-Leverage Knobs

### 1. Training Steps

This is the first parameter to vary.

OmniVoice's shipped fine-tuning config uses:

```json
"steps": 5000
```

Our first run used only:

```python
TRAINING_STEPS = 1000
```

Because both train and eval loss were still decreasing, the next experiments should try longer runs before changing too many other settings.

Reasonable sweep:

```python
TRAINING_STEPS = 2000
TRAINING_STEPS = 3000
TRAINING_STEPS = 5000
```

Keep `SAVE_STEPS` and `EVAL_STEPS` frequent enough to compare curves and retain checkpoints. For shorter runs, ensure `SAVE_STEPS <= TRAINING_STEPS`.

### 2. Learning Rate

The current script uses:

```python
LEARNING_RATE = 2e-5
```

OmniVoice's fine-tuning configs use:

```json
"learning_rate": 1e-5
```

The repository docs also describe fine-tuning as using a lower learning rate than training from scratch. Our `2e-5` run did not obviously diverge, but the loss curve is noisy and the learning rate decays to zero by the end of the configured steps.

Reasonable sweep:

```python
LEARNING_RATE = 5e-6
LEARNING_RATE = 1e-5
LEARNING_RATE = 2e-5
```

A longer run at `1e-5` is probably the cleanest next baseline because it follows the repository fine-tuning config more closely.

### 3. Effective Batch Size

OmniVoice docs identify `batch_tokens` as the primary memory control.

Repository fine-tuning config:

```json
"batch_tokens": 8192,
"gradient_accumulation_steps": 1
```

Current Kaggle config:

```python
BATCH_TOKENS = 1024
GRADIENT_ACCUMULATION_STEPS = 8
```

This is a practical T4 adaptation. It keeps a similar effective token budget while reducing VRAM pressure per forward pass.

Default recommendation: leave this alone unless training is unstable, too slow, or there is clear VRAM headroom.

Possible T4 experiments if memory allows:

```python
BATCH_TOKENS = 1536
GRADIENT_ACCUMULATION_STEPS = 6
```

or:

```python
BATCH_TOKENS = 2048
GRADIENT_ACCUMULATION_STEPS = 4
```

These preserve a similar effective batch size but may improve throughput. They can also cause OOM on Kaggle T4, so treat them as secondary experiments.

### 4. SDPA Length and Batch Limits

Because the Kaggle script uses:

```json
"attn_implementation": "sdpa"
```

these fields matter:

```python
MAX_SAMPLE_TOKENS = 1200
MAX_BATCH_SIZE = 4
```

The repository SDPA fine-tuning config uses:

```json
"max_sample_tokens": 2000,
"min_sample_tokens": 50,
"max_batch_size": 64
```

> Note: `max_sample_tokens` is a processed sequence-length filter, not a direct raw WAV duration limit. In OmniVoice's SDPA path, the sample length includes style/text tokens plus audio tokens after preprocessing. Our tokenized Filipino PLD shards are already far below the current `MAX_SAMPLE_TOKENS = 1200` audio-token range, so raising this to `1600` or `2000` is unlikely to be the main quality lever for the current dataset. Keep the main tuning focus on `TRAINING_STEPS` and `LEARNING_RATE` first.

Our T4-friendly values are more conservative than the official SDPA config, but `MAX_SAMPLE_TOKENS = 1200` is probably already permissive for the current PLD token shards.

After the main learning-rate and step-count runs, an optional alignment check is:

```python
MAX_SAMPLE_TOKENS = 1600
```

Then, if stable:

```python
MAX_SAMPLE_TOKENS = 2000
```

Be more cautious with:

```python
MAX_BATCH_SIZE
```

Raising it can improve throughput for many short samples, but it can also create memory spikes. Try `8` before anything larger.

## Data Considerations

PLD contains many very short word or name samples. For voice cloning quality, sentence-length Filipino utterances are likely more useful than isolated words.

If loss improves but generated speech remains poor, the next useful direction may not be only more steps. It may be better to adjust the dataset subset or weighting so training emphasizes cleaner and longer Filipino sentence-level utterances.

Possible future data-side experiments:

- train on a filtered subset with fewer very short utterances;
- increase representation of sentence-length samples;
- inspect whether English or named-entity-heavy lines are dominating some shards;
- compare audio quality from fixed inference prompts across runs, not just eval loss.

## Suggested Experiment Order

### Run A: Repository-Aligned Longer Baseline

```python
TRAINING_STEPS = 3000
LEARNING_RATE = 1e-5
```

Purpose: closer to OmniVoice's fine-tuning defaults while staying shorter than the full 5000-step config.

### Run B: Current Learning Rate, Longer Run

```python
TRAINING_STEPS = 3000
LEARNING_RATE = 2e-5
```

Purpose: test whether the current learning rate continues improving when given more steps.

### Run C: Include Longer Samples If Memory Allows

```python
TRAINING_STEPS = 3000
LEARNING_RATE = 1e-5
MAX_SAMPLE_TOKENS = 1600
```

Purpose: optional sanity check against the official SDPA length limit. This is lower priority than comparing training steps and learning rate because the current PLD token shards do not appear to be near the `1200` sample-token cap.

## Evaluation Notes

Do not rely only on `train/loss` and `eval/loss`.

For this project, keep the same fixed inference reference WAVs and prompts across all runs, then compare generated base and fine-tuned audio in W&B. The important question is whether the fine-tuned model improves:

- Filipino pronunciation;
- intelligibility;
- naturalness;
- speaker similarity to the reference;
- stability across all selected test references.

A slightly lower eval loss is only useful if the generated audio improves under the same listening samples.
