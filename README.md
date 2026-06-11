# Filipino OmniVoice Fine-Tuning

This repository contains a course research project on Filipino voice cloning and text-to-speech. We fine-tune `k2-fsa/OmniVoice` on cleaned Filipino read speech from the UP-DSP Philippine Languages Database, then compare the fine-tuned checkpoints against the base OmniVoice model.

OmniVoice already supports Filipino through the `fil` language ID. The project asks a narrower question: does Filipino-specific fine-tuning improve generated Filipino speech in intelligibility, speaker similarity, or naturalness?

## Quick Links

- [Machine Learning course paper](documents/machine_learning/machine_learning_paper.pdf)
- [Current Trends final paper](documents/current_trends/paper/final_paper.pdf)
- [Final proposal slides](documents/current_trends/presentation/final_beamer.pdf)
- [OmniVoice Filipino showcase](https://omnivoice-showcase.duncanb013.workers.dev/)

## Current State

The cleaned dataset package is ready for Kaggle:

```text
data/pld_filipino_clean/
  wavs/
  train.jsonl
  dev.jsonl
  test.jsonl
```

The package contains 42,276 Filipino read-speech samples, or 39.772 hours total. Speakers are split across train, dev, and test so the evaluation set uses held-out speakers.

| Split | Samples | Hours | Speakers |
| --- | ---: | ---: | ---: |
| train | 33,448 | 31.902 | 103 |
| dev | 4,506 | 3.705 | 13 |
| test | 4,322 | 4.165 | 13 |
| total | 42,276 | 39.772 | 129 |

Filtering kept read or prompted Filipino speech only. It removed spontaneous prompts, transcripts with parentheses or digits, unreadable or empty WAV files, and clips outside the 1.0 to 15.0 second range.

## Final Results

The completed controlled comparison evaluates the base model against three fine-tuning runs that share the same 5000-step setup and the same checkpoint policy: the development set is evaluated at fixed intervals and the checkpoint with the lowest development loss is kept. Only the learning rate differs across the three runs, so metric differences are attributable mostly to learning rate. All rows use the full 4,322-utterance speaker-disjoint PLD Filipino test split. WER measures intelligibility (lower is better), SIM-o measures speaker similarity (higher is better), and UTMOS estimates naturalness (higher is better).

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint at step 5000 | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint at step 5000 | 21.96 | -0.59 | 0.605 | 3.60 |

The LR 1e-5 checkpoint (`omnivoice-filipino-full-checkpoint-4900`) is the strongest overall: it gives the largest WER reduction, 4.03 absolute points below the base model, while keeping SIM-o slightly above base. LR 2e-5 is very close in WER at 18.83% but trades away speaker similarity (SIM-o 0.583, below base). LR 5e-6 preserves speaker similarity best among the fine-tunes (SIM-o 0.605) but only modestly improves WER. The base model keeps the highest UTMOS at 3.64, with the fine-tunes close behind at 3.60-3.61. The result is therefore an intelligibility-versus-speaker-similarity/naturalness tradeoff rather than a single-metric win, with LR 1e-5 offering the best overall balance.

## How the Work Runs

Training and evaluation run in Kaggle because the project uses free T4 GPUs. The `.py` files in `notebooks/` are Kaggle notebook exports that keep cell markers for markdown and code cells.

The usual workflow is:

1. Upload `data/pld_filipino_clean/` as a Kaggle dataset.
2. Clone and install OmniVoice inside the Kaggle notebook.
3. Tokenize `train.jsonl` and `dev.jsonl` into OmniVoice WebDataset shards.
4. Fine-tune OmniVoice with Kaggle T4-friendly settings such as FP16, SDPA attention, smaller token batches, and W&B tracking.
5. Generate matched Filipino samples from the base and fine-tuned models.
6. Evaluate WER, speaker similarity, and UTMOS on the same test set.

## Repository Map

| Path | Purpose |
| --- | --- |
| `.OmniVoice/` | Local clone of the upstream OmniVoice repository used for reference. |
| `data/pld_filipino_clean/` | Cleaned Filipino dataset package for Kaggle upload. |
| `scripts/prepare_omnivoice_filipino.py` | Builds the cleaned OmniVoice JSONL dataset from PLD WAV and LOG files. |
| `notebooks/` | Kaggle notebook-style Python files for tokenization, training, and evaluation. |
| `finetuning_plan.md` | Main runbook for the OmniVoice Filipino fine-tuning workflow. |
| `hyperparameter_tuning.md` | Notes on training settings and the controlled learning-rate comparison. |
| `eval-2e-5-and-5e-6/` | Final controlled evaluation artifacts for the best-eval LR 2e-5 and LR 5e-6 checkpoints (metrics, logs, manifest, figure, audio-example export). |
| `progress/` | Run notes and metric tracking. |
| `documents/` | Paper, presentation, figures, and related course outputs. |
| `showcase/` | Audio comparison web app for listening to base vs fine-tuned outputs. |
| `references/` | Source papers used by the project. |

## Rebuilding the Dataset

After the raw PLD files are available locally, rebuild the cleaned package with:

```powershell
python scripts\prepare_omnivoice_filipino.py --package-dir data\pld_filipino_clean --package-mode hardlink
```

The cleaned package should stay standalone. It should contain only `wavs/`, `train.jsonl`, `dev.jsonl`, and `test.jsonl` so it can be uploaded directly to Kaggle.

## Data and Use Limits

The UP-DSP Philippine Languages Database is released under CC-BY-NC-4.0. This project should be treated as an academic research prototype, not a commercial Filipino voice cloning product. The dataset should not be used to identify speakers, and redistribution or commercial use requires permission from the dataset owners.
