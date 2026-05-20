# Filipino OmniVoice Fine-Tuning

This repository contains a course research project on Filipino voice cloning and text-to-speech. We fine-tune `k2-fsa/OmniVoice` on cleaned Filipino read speech from the UP-DSP Philippine Languages Database, then compare the fine-tuned checkpoints against the base OmniVoice model.

OmniVoice already supports Filipino through the `fil` language ID. The project asks a narrower question: does Filipino-specific fine-tuning improve generated Filipino speech in intelligibility, speaker similarity, or naturalness?

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

## Results So Far

The strongest current checkpoint is `omnivoice-filipino-full-checkpoint-4900`. It lowers WER from 22.55% to 18.52% on the full test set, while keeping speaker similarity slightly above the base model. The base model still has the highest UTMOS score, so the current result is an intelligibility gain with a small naturalness tradeoff.

| Model | Checkpoint / artifact | WER (%) | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: |
| Base OmniVoice | `k2-fsa/OmniVoice` | 22.55 | 0.602 | 3.64 |
| Fine-tuned 1000 | `omnivoice-filipino-full-checkpoint-1000` | 20.07 | 0.610 | 3.60 |
| Fine-tuned 2000 | `omnivoice-filipino-full-checkpoint-2000` | 22.64 | 0.611 | 3.57 |
| Fine-tuned 4900 | `omnivoice-filipino-full-checkpoint-4900` | 18.52 | 0.604 | 3.61 |

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
| `hyperparameter_tuning.md` | Notes on training settings and checkpoint comparisons. |
| `progress/` | Run notes and metric tracking. |
| `documents/` | Paper, presentation, figures, and related course outputs. |
| `references/` | Source papers used by the project. |

## Rebuilding the Dataset

After the raw PLD files are available locally, rebuild the cleaned package with:

```powershell
python scripts\prepare_omnivoice_filipino.py --package-dir data\pld_filipino_clean --package-mode hardlink
```

The cleaned package should stay standalone. It should contain only `wavs/`, `train.jsonl`, `dev.jsonl`, and `test.jsonl` so it can be uploaded directly to Kaggle.

## Data and Use Limits

The UP-DSP Philippine Languages Database is released under CC-BY-NC-4.0. This project should be treated as an academic research prototype, not a commercial Filipino voice cloning product. The dataset should not be used to identify speakers, and redistribution or commercial use requires permission from the dataset owners.
