# Final Controlled Evaluation: Best-Eval LR 2e-5 and LR 5e-6

This folder holds the exported artifacts of the controlled new-learning-rate
evaluation run (`full-filipino-pld-test-new-lr-rerun-eval`, W&B eval run
`omnivoice-fil-new-lr-eval-892960a1`). It evaluated two of the three
controlled fine-tunes — the best-development-loss checkpoints of the 5000-step
LR `2e-5` and LR `5e-6` runs, both selected at step 5000 — on the full
4,322-utterance PLD Filipino test split, completing the final four-model
comparison. The evaluation notebook is
`../notebooks/omnivoice_evaluation_metrics_new_lr_reruns.py`.

Evaluated W&B checkpoint artifacts:

- LR `2e-5`: `omnivoice-full-filipino-pld-best-eval-lr-2e-5-checkpoint-best:v0`
- LR `5e-6`: `omnivoice-full-filipino-pld-best-eval-lr-5e-6-checkpoint-best:v0`

The base row in the summaries is reused from the first full evaluation run
(`../notebooks/omnivoice_evaluation_metrics.py`); the LR `1e-5` row of the
final comparison comes from the earlier
`../notebooks/omnivoice_evaluation_metrics_finetunes.py` run. The evaluation
setup (test split, manifests, inference settings, evaluator models, text
normalization) was identical across all of these runs.

## Contents

| Path | What it is |
| --- | --- |
| `metrics/metric_summary.json` | Parsed metric rows (WER with edit counts, SIM-o, UTMOS) for base reference, LR 2e-5, LR 5e-6. |
| `metrics/metric_summary.csv` | Same table, spreadsheet-friendly. |
| `metrics/metric_summary.md` | Same table, human-readable, with artifact notes. |
| `metric_logs/` | Raw WER, SIM-o, and UTMOS evaluator logs per new checkpoint. |
| `figures/metric_comparison.png` | The run's metric figure (base reference + the two new checkpoints; the four-model figure is `../documents/images/final_four_model_metric_comparison.png`). |
| `manifests/pld_filipino_full_test_eval.jsonl` | The evaluation manifest used for this run (4,322 rows). |
| `run_metadata.json` | Run name, W&B run ID, targets, and which heavy runtime dirs were not exported. |
| `run-omnivoice-fil-new-lr-eval-892960a1-…_v0/` | Raw W&B `finetune_evaluation_audio_examples` table export: 12 generated examples per new checkpoint, with reference prompts. Kept local (gitignored), like the other raw W&B exports. |

## Where the final audio examples live

The curated, reader-facing audio examples for all four final systems are in
`../showcase/public/audio/`, one wav per utterance ID, 12 utterances per
system:

| Showcase folder | System | Source export |
| --- | --- | --- |
| `base/` | Base OmniVoice | `../step-1000/evaluation_audio_examples.table.json` (`base` rows; raw export, gitignored) |
| `finetune_lr_1e-5/` | Best-eval LR 1e-5 (step 4900) | `../step-2000-and-best-eval-4900/finetune_evaluation_audio_examples.table.json` (`finetuned_best` rows; raw export, gitignored) |
| `finetune_lr_2e-5/` | Best-eval LR 2e-5 | this folder's table export (`best_eval_lr_2e_5` rows) |
| `finetune_lr_5e-6/` | Best-eval LR 5e-6 | this folder's table export (`best_eval_lr_5e_6` rows) |

`../showcase/public/audio/prompt/` holds the voice prompts the models heard
and `../showcase/public/audio/reference/` the ground-truth recordings; both
come from the same W&B tables/dataset and are shared across systems.

## Final four-model results

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint at step 5000 | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint at step 5000 | 21.96 | -0.59 | 0.605 | 3.60 |

See `../progress/2026-06-11-controlled-new-lr-eval.md` for the full write-up.
