# Map of Content

This project studies Filipino fine-tuning of OmniVoice using the Filipino subset
of the UP-DSP Philippine Languages Database. Training and evaluation were run in
Kaggle, with local files kept as notebook-style scripts, course outputs, data
packages, and reference material.

## Start Here

`readme.md` gives the project frame, research question, dataset choice, and
current objective results.

`finetuning_plan.md` is the main runbook. It describes the dataset, Kaggle setup,
training flow, evaluation flow, and current result table.

`progress/2026-06-11-controlled-new-lr-eval.md` is the clearest short record
of the final controlled learning-rate comparison. The strongest overall
checkpoint is the best-eval LR 1e-5 model,
`omnivoice-filipino-full-checkpoint-4900`.

## Root Project Docs

- `AGENTS.md` contains local project instructions for Codex.
- `readme.md` is the project overview.
- `task.md` is the implementation guide and definition of done.
- `finetuning_plan.md` is the Kaggle training and evaluation runbook.
- `hyperparameter_tuning.md` records tuning decisions and observed results.
- `training_concepts.md` explains OmniVoice training concepts and T4 constraints.

## Course Documents

Course outputs now live under `documents/` so the root stays focused on project
setup, data, notebooks, scripts, references, and the upstream repository.

### `documents/current_trends/`

This folder contains Current / Computing Trends outputs.

- `documents/current_trends/presentation/` contains the proposal slide content,
  Beamer source, compiled proposal deck, and presentation README.
- `documents/current_trends/paper/` contains the proposal paper, final paper,
  compiled PDFs, plain text extraction, and paper README.

### `documents/machine_learning/`

This folder contains the separate Machine Learning course paper.

- `documents/machine_learning/machine_learning_paper.tex` is the source.
- `documents/machine_learning/machine_learning_paper.pdf` is the compiled paper.
- `documents/machine_learning/README.md` explains the folder.

### `documents/images/`

This shared image folder contains screenshots and figures used by both course
outputs. It includes OmniVoice paper excerpts, PLD paper excerpts, the Mozilla
Data Collective page, and app screenshots used in the proposal presentation.

## Kaggle Notebooks

`notebooks/` contains Kaggle notebook-style `.py` files with `# %%` cells. These
mirror the scripts pasted into Kaggle notebooks.

- `notebooks/omnivoice_training.py` produced the early exploratory 1000-step
  checkpoint.
- `notebooks/omnivoice_training_best_eval.py` is the controlled 5000-step
  best-development-loss run at LR `1e-5`.
- `notebooks/omnivoice_training_best_eval_lr_2e_5.py` and
  `notebooks/omnivoice_training_best_eval_lr_5e_6.py` repeat the same
  controlled setup at LR `2e-5` and LR `5e-6`.
- `notebooks/tokenize_dataset.py` handles tokenization in Kaggle.
- `notebooks/omnivoice_evaluation_metrics.py` evaluates the base model and the
  first fine-tuned checkpoint.
- `notebooks/omnivoice_evaluation_metrics_finetunes.py` evaluates later
  fine-tuned checkpoints and reuses the base metric row.
- `notebooks/omnivoice_evaluation_metrics_new_lr_reruns.py` evaluates the
  controlled best-eval LR `2e-5` and LR `5e-6` checkpoints; its exported
  artifacts live in `eval-2e-5-and-5e-6/`.
- `notebooks/smoke_evaluation_metrics.py` is for smaller evaluation checks.
- `notebooks/README.md` gives a short notebook summary and metric table.

## Data

### `data/`

This folder holds Kaggle-ready datasets used by the project.

- `data/pld_filipino_clean/` is the cleaned Filipino PLD dataset with
  `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `wavs/`.
- `data/pld_filipino_clean.zip` is the zipped cleaned upload package.
- `data/omnivoice_pld_audio_tokens_kaggle_safe/` is the Kaggle-safe tokenized
  OmniVoice audio-token dataset.
- `data/omnivoice_pld_audio_tokens_kaggle_safe.zip` is the zipped tokenized
  upload package.
- `data/README.md` explains the folder.

### `data_artifacts/`

This folder holds source, raw, bulky, or intermediate data artifacts that are
not the final Kaggle-ready dataset folder.

- `data_artifacts/original_pld/` is the original extracted Filipino PLD WAV/LOG
  tree.
- `data_artifacts/filipino_omnivoice_manifests/` is an earlier local
  OmniVoice-style manifest output.
- `data_artifacts/omnivoice_filipino_pld_tokens_full_original_tar/` is the
  original tokenized dataset with `.tar` WebDataset shards. It has the same
  train/dev token counts as the Kaggle-safe package, but Kaggle-safe upload uses
  the `.wds` repackaged version under `data/`.
- `data_artifacts/README.md` explains the folder.

## Final Evaluation Artifacts

`eval-2e-5-and-5e-6/` holds the exported artifacts of the controlled
best-eval LR `2e-5` and LR `5e-6` evaluation run
(`full-filipino-pld-test-new-lr-rerun-eval`):

- `eval-2e-5-and-5e-6/metrics/` has the metric summary as JSON, CSV, and
  markdown.
- `eval-2e-5-and-5e-6/metric_logs/` has the raw WER, SIM-o, and UTMOS logs.
- `eval-2e-5-and-5e-6/figures/metric_comparison.png` is the run's metric
  figure.
- `eval-2e-5-and-5e-6/manifests/pld_filipino_full_test_eval.jsonl` is the
  evaluation manifest.
- `eval-2e-5-and-5e-6/README.md` documents provenance and maps the audio
  examples for all four final systems.

## Showcase Web App

`showcase/` is the deployed audio comparison app. It plays matched voice-prompt,
ground-truth, and generated audio for the base model and the three best-eval
fine-tunes on 12 test utterances, with the final metric table.

## Scripts

`scripts/` contains local helper scripts.

- `scripts/prepare_omnivoice_filipino.py` prepares
  `data_artifacts/original_pld/` into `data/pld_filipino_clean/`.
- `scripts/repackage_omnivoice_tokens_for_kaggle.py` repackages the original
  tokenized output into `data/omnivoice_pld_audio_tokens_kaggle_safe/`.
- `scripts/README.md` explains the folder.

## Progress

`progress/` contains dated progress notes.

- `progress/2026-05-18-omnivoice-kaggle-smoke.md` records early Kaggle
  smoke-test progress.
- `progress/2026-05-19-full-train-track-metrics.md` records training and
  evaluation progress after the first full metric runs (now project history).
- `progress/2026-06-11-showcase-web-app.md` records the audio comparison
  showcase web app build.
- `progress/2026-06-11-controlled-new-lr-eval.md` records the completed
  controlled learning-rate evaluation and the final four-model comparison.
- `progress/README.md` explains the folder.

## References and Upstream Code

### `references/`

- `references/Zhu et al - OmniVoice.pdf` is the OmniVoice paper.
- `references/Cajote et al - Philippine Languages Database.pdf` is the PLD
  paper.
- `references/README.md` explains the folder.

### `.OmniVoice/`

`.OmniVoice/` is the cloned upstream OmniVoice repository. It is hidden at the
root because it is external reference code, not project-authored code.

Important OmniVoice reference files:

- `.OmniVoice/README.md`
- `.OmniVoice/examples/README.md`
- `.OmniVoice/docs/data_preparation.md`
- `.OmniVoice/docs/training.md`
- `.OmniVoice/docs/evaluation.md`
- `.OmniVoice/examples/run_finetune.sh`

## Results

The final controlled comparison evaluates the base model against three 5000-step
best-development-loss fine-tuning runs that differ only in learning rate:

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint | 21.96 | -0.59 | 0.605 | 3.60 |

The strongest overall model is the best-eval LR `1e-5` checkpoint
(`omnivoice-filipino-full-checkpoint-4900`). It has the lowest WER, reducing the
base model from 22.55% to 18.52%, with SIM-o slightly above base. LR `2e-5` is
nearly as intelligible (18.83% WER) but loses speaker similarity (SIM-o 0.583),
and LR `5e-6` keeps the best fine-tune SIM-o (0.605) with only a modest WER
gain. The base model has the highest UTMOS (3.64), so the overall finding is an
intelligibility-versus-speaker-similarity/naturalness tradeoff. Earlier
exploratory final-step checkpoints (1000-step `2e-5`, 2000-step `5e-6`) are
project history, not part of this comparison.

## Generated and Cache Files

LaTeX build files such as `.aux`, `.log`, and `.out` appear in the document
folders after compilation.

`__pycache__/` and `notebooks/__pycache__/` are Python cache folders. They are
not part of the research content.

`omnivoice.ipynb` is currently empty. The useful Kaggle notebook records are the
`.py` files in `notebooks/`.

## Reading Order

Read `readme.md` first if you need the project frame.

Read `documents/current_trends/paper/final_paper.pdf` if you need the Current /
Computing Trends submission.

Read `documents/machine_learning/machine_learning_paper.pdf` if you need the
Machine Learning submission.

Read `finetuning_plan.md` if you need to reproduce the workflow.

Read `notebooks/README.md` and the relevant notebook-style `.py` file if you
need to rerun training or evaluation in Kaggle.

Read `progress/2026-06-11-controlled-new-lr-eval.md` if you only need the
final metric comparison and best-model decision.

Open `showcase/` (or the deployed app) if you want to listen to base versus
fine-tuned outputs directly.
