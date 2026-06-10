# OmniVoice Filipino Hyperparameter Tuning Notes

These notes summarize what to tune after the first full Filipino PLD fine-tuning run on Kaggle. The 1000-step run showed both `train/loss` and `eval/loss` still trending downward, with eval loss ending near its best value. That does not look like clear overfitting yet. It more likely means the run is still short.

The guidance below is grounded in OmniVoice's own fine-tuning configs and docs, especially:

- `.OmniVoice/examples/config/train_config_finetune.json`
- `.OmniVoice/examples/config/train_config_finetune_sdpa.json`
- `.OmniVoice/docs/training.md`

## Observed Results After Later Runs

After the initial 1000-step run, two more fine-tuned checkpoints were evaluated on the same full PLD Filipino test split:

- `omnivoice-filipino-full-checkpoint-2000`: 2000 steps, learning rate `5e-6`.
- `omnivoice-filipino-full-checkpoint-4900`: 5000-step run with the checkpoint selected by lowest development loss, learning rate `1e-5`, selected step 4900.

| Model | Learning rate | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | N/A | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Fine-tuned 1000 | `2e-5` | 1000 steps | 20.07 | -2.48 | 0.610 | 3.60 |
| Fine-tuned 2000 | `5e-6` | 2000 steps | 22.64 | +0.09 | 0.611 | 3.57 |
| Fine-tuned 4900 | `1e-5` | lowest eval loss at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |

Current tuning conclusion: the `1e-5` longer run with development-loss checkpointing is the strongest model so far. It gives the lowest WER, improving the base model by 4.03 absolute points, while preserving SIM-o slightly above base. The `5e-6` 2000-step run is not better overall: it has the highest SIM-o but loses the WER gain and drops UTMOS.

This means lower learning rate alone was not enough. The useful change appears to be the combination of a repository-aligned `1e-5` learning rate, longer training, and checkpoint selection by development loss.

However, the current table is not a clean learning-rate comparison because the 1000-step and 2000-step runs used final-step checkpoints, while the 4900-step result came from a 5000-step run selected by lowest development loss. For the paper discussion, treat the table above as project history. The cleaner comparison should rerun the same 5000-step best-development-loss checkpointing setup at:

| Controlled run | Learning rate | Steps | Checkpoint policy | Notebook |
| --- | ---: | ---: | --- | --- |
| Best-eval LR 1e-5 | `1e-5` | 5000 | lowest dev loss | `notebooks/omnivoice_training_best_eval.py` |
| Best-eval LR 2e-5 | `2e-5` | 5000 | lowest dev loss | `notebooks/omnivoice_training_best_eval_lr_2e_5.py` |
| Best-eval LR 5e-6 | `5e-6` | 5000 | lowest dev loss | `notebooks/omnivoice_training_best_eval_lr_5e_6.py` |

After these reruns, compare WER, SIM-o, and UTMOS only across the selected best checkpoints. That will let the discussion attribute differences mostly to learning rate instead of checkpoint policy or training duration.

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
