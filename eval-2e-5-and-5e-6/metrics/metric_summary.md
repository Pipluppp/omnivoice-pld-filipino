# OmniVoice Filipino new LR rerun evaluation

Run: `full-filipino-pld-test-new-lr-rerun-eval`

This summary evaluates only the two new controlled best-eval learning-rate reruns. The base row is reused from the previous full-test evaluation for reference.

| Model | Source | WER (%) | SIM-o | UTMOS | Artifact / notes |
|---|---:|---:|---:|---:|---|
| base_reference | previous_full_eval | 22.55 | 0.602 | 3.64 | Base model metrics reused from omnivoice_evaluation_metrics.py full run. |
| best_eval_lr_2e_5 | current_eval | 18.83 | 0.583 | 3.61 | duncanb013-polytechnic-university-of-the-philippines/OmniVoice-PLD-Filipino/omnivoice-full-filipino-pld-best-eval-lr-2e-5-checkpoint-best:v0 |
| best_eval_lr_5e_6 | current_eval | 21.96 | 0.605 | 3.60 | duncanb013-polytechnic-university-of-the-philippines/OmniVoice-PLD-Filipino/omnivoice-full-filipino-pld-best-eval-lr-5e-6-checkpoint-best:v0 |

Included files:

- `metric_summary.json`: parsed metric rows.
- `metric_summary.csv`: spreadsheet-friendly metric table.
- `metric_logs/`: raw WER, SIM-o, and UTMOS logs for each new checkpoint.
- `figures/metric_comparison.png`: compact visual comparison.
- `pld_filipino_full_test_eval.jsonl`: evaluation manifest used for this run.
