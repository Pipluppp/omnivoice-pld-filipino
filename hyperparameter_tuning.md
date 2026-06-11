# OmniVoice Filipino Hyperparameter Tuning Notes

These notes record the training settings of the Filipino PLD fine-tuning runs on Kaggle and the completed controlled learning-rate comparison. The guidance is grounded in OmniVoice's own fine-tuning configs and docs, especially:

- `.OmniVoice/examples/config/train_config_finetune.json`
- `.OmniVoice/examples/config/train_config_finetune_sdpa.json`
- `.OmniVoice/docs/training.md`

## Final Controlled Learning-Rate Comparison

All three runs used the identical 5000-step setup and the same checkpoint policy (evaluate the development set at fixed intervals, save only when development loss improves), differing only in learning rate:

| Controlled run | Learning rate | Steps | Best checkpoint | Training notebook |
| --- | ---: | ---: | --- | --- |
| Best-eval LR 1e-5 | `1e-5` | 5000 | lowest dev loss at step 4900 | `notebooks/omnivoice_training_best_eval.py` |
| Best-eval LR 2e-5 | `2e-5` | 5000 | lowest dev loss at step 5000 | `notebooks/omnivoice_training_best_eval_lr_2e_5.py` |
| Best-eval LR 5e-6 | `5e-6` | 5000 | lowest dev loss at step 5000 | `notebooks/omnivoice_training_best_eval_lr_5e_6.py` |

All four systems were evaluated on the same full 4,322-utterance PLD Filipino test split with the same inference settings, evaluator models, and text normalization. The base row comes from `notebooks/omnivoice_evaluation_metrics.py`, the LR 1e-5 row from `notebooks/omnivoice_evaluation_metrics_finetunes.py`, and the LR 2e-5 and LR 5e-6 rows from `notebooks/omnivoice_evaluation_metrics_new_lr_reruns.py`, exported locally to `eval-2e-5-and-5e-6/`.

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint at step 5000 | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint at step 5000 | 21.96 | -0.59 | 0.605 | 3.60 |

Tuning conclusions:

- `1e-5` is the strongest learning rate overall by WER: 18.52%, a 4.03 absolute-point reduction versus base, while keeping SIM-o slightly above base.
- `2e-5` is very close in WER at 18.83% but trades away speaker similarity: its SIM-o of 0.583 is the only fine-tune value below base.
- `5e-6` is the most conservative update: it has the best SIM-o among the fine-tunes at 0.605 but only a modest 0.59-point WER improvement over base.
- The base model keeps the highest UTMOS at 3.64; the fine-tunes sit close together at 3.60-3.61.

Because the checkpoint policy and step count are held constant, these differences are attributable mostly to learning rate. The pattern is an intelligibility-versus-speaker-similarity/naturalness tradeoff: a larger learning rate moves the model further toward Filipino intelligibility at some cost to speaker similarity, while a smaller learning rate stays closer to the base model on every axis.

## Kaggle Training Configuration

Each controlled run uses the same T4-friendly SDPA setup; only `LEARNING_RATE` differs:

```python
TRAINING_STEPS = 5000
LEARNING_RATE = 1e-5  # 5e-6 and 2e-5 in the other two runs
BATCH_TOKENS = 1024
GRADIENT_ACCUMULATION_STEPS = 8
MAX_BATCH_SIZE = 4
MAX_SAMPLE_TOKENS = 1200
EVAL_STEPS = 250
```

There is no fixed `SAVE_STEPS`; a checkpoint is saved whenever the development loss improves at an evaluation, so each run ends with its best-development-loss checkpoint.

The effective token budget per optimizer update is roughly:

```text
BATCH_TOKENS * GRADIENT_ACCUMULATION_STEPS = 1024 * 8 = 8192 tokens
```

This matches the repository fine-tuning config's `batch_tokens = 8192`, but splits it into smaller per-forward batches so it can fit Kaggle's free T4 GPU.

## Setting Rationale

### Training Steps

The runs use the repository fine-tuning config's step count:

```json
"steps": 5000
```

With `EVAL_STEPS = 250`, each run gets 20 development evaluations, and the best-development-loss checkpoint policy turns the long run into a principled checkpoint search rather than a bet on the final step.

### Learning Rate

The swept values bracket the repository default:

```python
LEARNING_RATE = 5e-6
LEARNING_RATE = 1e-5
LEARNING_RATE = 2e-5
```

OmniVoice's fine-tuning configs use `1e-5`, and the repository docs describe fine-tuning as using a lower learning rate than training from scratch. The sweep tests one value below and one above that default; the learning rate decays to zero by the end of the configured steps under the cosine schedule.

### Effective Batch Size

OmniVoice docs identify `batch_tokens` as the primary memory control.

Repository fine-tuning config:

```json
"batch_tokens": 8192,
"gradient_accumulation_steps": 1
```

Kaggle config:

```python
BATCH_TOKENS = 1024
GRADIENT_ACCUMULATION_STEPS = 8
```

This is a practical T4 adaptation. It keeps the same effective token budget while reducing VRAM pressure per forward pass. Leave it alone unless training is unstable, too slow, or there is clear VRAM headroom; raising `BATCH_TOKENS` can cause OOM on Kaggle T4.

### SDPA Length and Batch Limits

Because the runs use:

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

> Note: `max_sample_tokens` is a processed sequence-length filter, not a direct raw WAV duration limit. In OmniVoice's SDPA path, the sample length includes style/text tokens plus audio tokens after preprocessing. The tokenized Filipino PLD shards are already far below the `MAX_SAMPLE_TOKENS = 1200` audio-token range, so raising this is unlikely to be a quality lever for the current dataset.

The T4-friendly values are more conservative than the official SDPA config. Raising `MAX_BATCH_SIZE` can improve throughput for many short samples, but it can also create memory spikes.

## Data Considerations

PLD contains many very short word or name samples. For voice cloning quality, sentence-length Filipino utterances are likely more useful than isolated words.

If loss improves but generated speech remains poor, the next useful direction may not be only more steps. It may be better to adjust the dataset subset or weighting so training emphasizes cleaner and longer Filipino sentence-level utterances.

Possible future data-side experiments:

- train on a filtered subset with fewer very short utterances;
- increase representation of sentence-length samples;
- inspect whether English or named-entity-heavy lines are dominating some shards;
- compare audio quality from fixed inference prompts across runs, not just eval loss.

## Evaluation Notes

Do not rely only on `train/loss` and `eval/loss`.

For this project, keep the same fixed inference reference WAVs and prompts across all runs, then compare generated base and fine-tuned audio in W&B. The important question is whether the fine-tuned model improves:

- Filipino pronunciation;
- intelligibility;
- naturalness;
- speaker similarity to the reference;
- stability across all selected test references.

A slightly lower eval loss is only useful if the generated audio improves under the same listening samples. The final comparison shows why: the LR 1e-5 and 2e-5 checkpoints are nearly tied in development loss and WER, yet differ visibly in SIM-o, and the LR 5e-6 run's insertion increase is invisible in its loss curve.
