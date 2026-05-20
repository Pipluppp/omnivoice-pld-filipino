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

`progress/2026-05-19-full-train-track-metrics.md` is the clearest short record
of the final metric comparison. It states the current best checkpoint:
`omnivoice-filipino-full-checkpoint-best`.

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

- `notebooks/omnivoice_training.py` produced the 1000-step checkpoint.
- `notebooks/omnivoice_training_best_eval.py` saved the best checkpoint by
  development loss.
- `notebooks/tokenize_dataset.py` handles tokenization in Kaggle.
- `notebooks/omnivoice_evaluation_metrics.py` evaluates the base model and the
  first fine-tuned checkpoint.
- `notebooks/omnivoice_evaluation_metrics_finetunes.py` evaluates later
  fine-tuned checkpoints and reuses the base metric row.
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
  evaluation progress after the full metric runs.
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

The main evaluated systems are:

| Model | Checkpoint / artifact | Learning rate | Selection | WER (%) | SIM-o | UTMOS |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Base OmniVoice | `k2-fsa/OmniVoice` | N/A | pretrained base | 22.55 | 0.602 | 3.64 |
| Fine-tuned 1000 | `omnivoice-filipino-full-checkpoint-1000` | `2e-5` | 1000 steps | 20.07 | 0.610 | 3.60 |
| Fine-tuned 2000 | `omnivoice-filipino-full-checkpoint-2000` | `5e-6` | 2000 steps | 22.64 | 0.611 | 3.57 |
| Fine-tuned best | `omnivoice-filipino-full-checkpoint-best` | `1e-5` | best eval loss at step 4900 | 18.52 | 0.604 | 3.61 |

The current best model is `omnivoice-filipino-full-checkpoint-best`. It has the
lowest WER, reducing the base model from 22.55% to 18.52%. It does not have the
highest SIM-o or UTMOS, but SIM-o stays slightly above the base model and UTMOS
stays close to the other fine-tuned checkpoints.

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

Read `progress/2026-05-19-full-train-track-metrics.md` if you only need the
final metric comparison and best-model decision.
