# Images

This folder contains the figures and screenshots used by the papers and slides.

## Final results figures

- `final_four_model_metric_comparison.png` is the publication-style figure for
  the final controlled comparison: WER, SIM-o, and UTMOS for the base model
  and the three 5000-step best-development-loss fine-tunes (LR 1e-5, 2e-5,
  5e-6) on the full PLD Filipino test split.
- `new_lr_eval_run_metric_comparison.png` is the metric figure exported by the
  LR 2e-5 / LR 5e-6 evaluation run, copied from
  `../../eval-2e-5-and-5e-6/figures/metric_comparison.png`. It covers only the
  two new checkpoints plus the base reference row.

## Reference and dataset excerpts

- `omnivoice paper.png` — OmniVoice paper first page.
- `omnivoice filipino dataset.png` — OmniVoice Filipino training-hours excerpt.
- `philippine languages database paper.png` — PLD paper excerpt.
- `philippine languages database paper list of language.png` — PLD language
  statistics table excerpt.
- `up dsp philippine languages database.png` — Mozilla Data Collective dataset
  page.

## Training run records (W&B screenshots)

- `finetunes_training_loss_together.png` — combined training-loss curves for
  the fine-tuning runs.
- `step 5000 lr 1e-5.png` — 5000-step LR 1e-5 best-eval run (part of the final
  controlled comparison; its selected checkpoint is step 4900).
- `step 1000 lr 2e-5.png` — early exploratory 1000-step LR 2e-5 run
  (project history; superseded by the controlled best-eval LR 2e-5 run).
- `step 2000 lr 5e-6.png` — early exploratory 2000-step LR 5e-6 run
  (project history; superseded by the controlled best-eval LR 5e-6 run).

## Generated-sample tables (W&B screenshots)

- `base_test_samples.png` — base model generated-sample table.
- `finetune_5000_test_samples.png` — 5000-step LR 1e-5 run sample table.
- `finetune_1000_test_samples.png` — exploratory 1000-step run sample table
  (project history).
- `finetune_2000_test_samples.png` — exploratory 2000-step run sample table
  (project history).

## App screenshots

- `app 1.png`, `app 2.png`, `app 3.png` — voice cloning application
  screenshots used in the proposal presentation.
