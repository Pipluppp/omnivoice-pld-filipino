# 2026-06-11 controlled new-LR evaluation complete

The controlled best-eval learning-rate comparison is finished. The two
remaining 5000-step best-development-loss runs, at LR `2e-5`
([omnivoice_training_best_eval_lr_2e_5.py](../notebooks/omnivoice_training_best_eval_lr_2e_5.py))
and LR `5e-6`
([omnivoice_training_best_eval_lr_5e_6.py](../notebooks/omnivoice_training_best_eval_lr_5e_6.py)),
were evaluated on the full PLD Filipino test split with
[omnivoice_evaluation_metrics_new_lr_reruns.py](../notebooks/omnivoice_evaluation_metrics_new_lr_reruns.py)
(run `full-filipino-pld-test-new-lr-rerun-eval`, W&B eval run
`omnivoice-fil-new-lr-eval-892960a1`, 4,322 eval samples). The base row was
reused from the first full evaluation; the evaluation setup (test split,
manifests, inference settings, evaluator models, text normalization) was
unchanged.

Evaluated W&B checkpoint artifacts:

- LR `2e-5`: `omnivoice-full-filipino-pld-best-eval-lr-2e-5-checkpoint-best:v0`
- LR `5e-6`: `omnivoice-full-filipino-pld-best-eval-lr-5e-6-checkpoint-best:v0`

Exported artifacts live in [`eval-2e-5-and-5e-6/`](../eval-2e-5-and-5e-6/):
metric summaries (`metrics/`), raw WER/SIM-o/UTMOS logs (`metric_logs/`), the
evaluation manifest (`manifests/pld_filipino_full_test_eval.jsonl`), the run
metric figure (`figures/metric_comparison.png`), and the 12-sample audio
example table export.

## Final four-model comparison

All fine-tunes share the same 5000-step setup and best-development-loss
checkpoint policy, so differences are attributable mostly to learning rate.

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint | 21.96 | -0.59 | 0.605 | 3.60 |

WER edit-operation counts:

| Model | Insertions | Deletions | Substitutions | Words |
| --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | 2,668 | 640 | 2,858 | 27,342 |
| Best-eval LR 1e-5 | 2,045 | 603 | 2,417 | 27,354 |
| Best-eval LR 2e-5 | 2,156 | 604 | 2,392 | 27,354 |
| Best-eval LR 5e-6 | 3,005 | 577 | 2,426 | 27,354 |

## Reading the results

- LR `1e-5` remains the strongest overall by WER: 18.52%, a 4.03
  absolute-point reduction versus base, with SIM-o slightly above base.
- LR `2e-5` is very close in WER (18.83%) but has the lowest SIM-o (0.583),
  the only fine-tune below base. It cuts insertions and substitutions like the
  `1e-5` run but pays in speaker similarity.
- LR `5e-6` keeps the best fine-tune SIM-o (0.605) but improves WER by only
  0.59 points; its insertions (3,005) stay above base, the same pattern the
  old 2000-step `5e-6` run showed.
- Base keeps the highest UTMOS (3.64); the fine-tunes are close at 3.60-3.61.

The pattern is an intelligibility-versus-speaker-similarity/naturalness
tradeoff, not a single-metric win: larger learning rates buy intelligibility,
smaller ones preserve the base model's speaker similarity. LR `1e-5` is the
best balance and stays the recommended checkpoint
(`omnivoice-filipino-full-checkpoint-4900`).

## Downstream updates done with this note

- Final tables updated across `readme.md`, `task.md`, `moc.md`,
  `hyperparameter_tuning.md`, `finetuning_plan.md`, `notebooks/README.md`.
- Papers and final presentation rewritten around the four-model controlled
  comparison; old 1000-step/2000-step results moved to project history.
- Showcase (`showcase/src/data/samples.ts`) metrics and labels updated to the
  controlled checkpoints; LR `2e-5` and `5e-6` audio replaced with exports
  from the new best-eval checkpoints.
