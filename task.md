# Implementation Task Guide

This document is the implementor-facing guide for carrying out the Filipino OmniVoice fine-tuning research project. It explains which project documents to follow, what artifacts to build, and the recommended order of work.

## Goal

Fine-tune the pretrained `k2-fsa/OmniVoice` model on curated Filipino read speech from the UP-DSP Philippine Languages Database, then compare the base and fine-tuned models using intelligibility, naturalness, and speaker-similarity metrics.

The research question is:

> How much does Filipino-specific fine-tuning improve OmniVoice performance for Filipino TTS and voice cloning across intelligibility, naturalness, and speaker similarity metrics?

## Primary Document to Follow

Use [`finetuning_plan.md`](finetuning_plan.md) as the main execution runbook.

It contains the practical plan for:

- preparing the Filipino dataset;
- converting WAV and transcript files into OmniVoice JSONL;
- splitting train/dev/test data;
- tokenizing audio into OmniVoice WebDataset shards;
- setting up Kaggle T4 training;
- configuring fine-tuning;
- generating base and fine-tuned samples;
- evaluating results.

## Supporting Documents

Use these documents as references while implementing.

| Document | Purpose |
| --- | --- |
| [`readme.md`](readme.md) | Project framing, research question, dataset choice, expected outputs, and success criteria. |
| [`.OmniVoice/README.md`](.OmniVoice/README.md) | OmniVoice installation, inference, CLI usage, and general model behavior. |
| [`.OmniVoice/examples/README.md`](.OmniVoice/examples/README.md) | Official training, fine-tuning, and evaluation example overview. |
| [`.OmniVoice/examples/run_finetune.sh`](.OmniVoice/examples/run_finetune.sh) | Official fine-tuning script template to adapt for Kaggle. |
| [`.OmniVoice/docs/data_preparation.md`](.OmniVoice/docs/data_preparation.md) | Required JSONL format and WebDataset/tokenization process. |
| [`.OmniVoice/docs/training.md`](.OmniVoice/docs/training.md) | Training config fields, SDPA attention, checkpointing, and `accelerate` usage. |
| [`.OmniVoice/docs/evaluation.md`](.OmniVoice/docs/evaluation.md) | WER/CER, speaker similarity, and UTMOS evaluation references. |

## Current Readiness

The local pre-Kaggle dataset preparation is complete.

The project is ready for the Kaggle training phase because:

- OmniVoice is open source and includes official fine-tuning scripts.
- OmniVoice already supports Filipino using language ID `fil`.
- The UP-DSP Philippine Languages Database provides Filipino speech and transcript files suitable for supervised TTS fine-tuning.
- The expected OmniVoice data format is documented.
- The evaluation metrics are defined and aligned with the OmniVoice paper.
- The Kaggle T4 constraints and likely config changes are already planned.
- The cleaned Filipino dataset now exists as a standalone upload-ready directory:

```text
data/pld_filipino_clean/
  wavs/
  train.jsonl
  dev.jsonl
  test.jsonl
```

Current split:

| Split | Samples | Percent | Hours | Speakers |
| --- | ---: | ---: | ---: | ---: |
| train | 33,448 | 79.12% | 31.902 | 103 |
| dev | 4,506 | 10.66% | 3.705 | 13 |
| test | 4,322 | 10.22% | 4.165 | 13 |
| total | 42,276 | 100% | 39.772 | 129 |

The remaining work is Kaggle/training-level:

- create Kaggle-specific config files;
- upload or attach `data/pld_filipino_clean/` as a Kaggle dataset;
- tokenize train/dev inside the Kaggle notebook;
- run a small training smoke test;
- adjust memory-related settings if Kaggle T4 runs out of VRAM;
- generate and evaluate samples.

## Implementation Order

### 1. Prepare the Dataset

Download the UP-DSP Philippine Languages Database from Mozilla Data Collective.

Use only the Filipino read-speech subset. Spontaneous speech has been excluded because it requires stricter transcript verification and is less suitable for this first supervised TTS fine-tuning run.

Implemented script:

```text
scripts/prepare_omnivoice_filipino.py
```

The script:

1. locate Filipino WAV files;
2. locate matching LOG/transcript files;
3. normalize transcript whitespace lightly;
4. remove missing, empty, unreadable, zero-byte, spontaneous, parenthetical, and digit-containing samples;
5. keep utterances from 1.0 to 15.0 seconds;
6. assign `language_id: "fil"`;
7. split data by speaker into train/dev/test;
8. write:

```text
data/pld_filipino_clean/train.jsonl
data/pld_filipino_clean/dev.jsonl
data/pld_filipino_clean/test.jsonl
```

Expected JSONL shape:

```json
{"id":"0000.110816.021250.0001","audio_path":"wavs/0000/0000.110816.021250.0001.wav","text":"manugang","language_id":"fil"}
```

Do not add metadata fields to `data/pld_filipino_clean`; it should remain a standalone upload directory with only WAV files and the OmniVoice JSONL manifests.

### 2. Verify the Dataset

Before tokenization, verify:

- audio paths exist;
- transcripts are non-empty;
- files are readable by `torchaudio` or `soundfile`;
- sample rate is acceptable or can be resampled;
- duration filtering works;
- train/dev/test speakers do not overlap.

Validated outputs include:

- number of utterances per split;
- total hours per split;
- number of speakers per split;
- min/median/max duration;
- examples of JSONL lines.
- speaker-disjoint train/dev/test IDs.

### 3. Prepare Kaggle Workspace

Use a Kaggle notebook with GPU acceleration enabled. Prefer `2x T4` if available.

Recommended Kaggle paths:

```text
/kaggle/input/pld-filipino-clean/  # uploaded data/pld_filipino_clean dataset
/kaggle/working/OmniVoice/         # cloned OmniVoice repo
/kaggle/working/data/              # tokenized shards and training manifests
/kaggle/working/exp/               # training outputs
/kaggle/working/results/           # generated samples and evaluation outputs
```

Clone and install OmniVoice:

```bash
cd /kaggle/working
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

If Kaggle dependency conflicts occur, resolve them before training. The main dependencies to watch are `torch`, `torchaudio`, `transformers`, `accelerate`, and `deepspeed`.

### 4. Tokenize Audio for OmniVoice

Tokenization happens inside Kaggle after `data/pld_filipino_clean/` is uploaded or attached. It should not be done before upload because the token shards are training artifacts and can be regenerated in `/kaggle/working`.

Use OmniVoice's data preparation flow from:

```text
.OmniVoice/docs/data_preparation.md
```

Tokenize `train.jsonl` and `dev.jsonl` into WebDataset shards.

The expected outputs are:

```text
/kaggle/working/data/finetune/tokens/train/data.lst
/kaggle/working/data/finetune/tokens/dev/data.lst
```

These `data.lst` files are what the OmniVoice training data config should reference.

### 5. Create Kaggle-Specific Configs

Create Kaggle-oriented config files instead of modifying the original examples directly.

Suggested files:

```text
.OmniVoice/examples/config/train_config_finetune_kaggle_t4.json
.OmniVoice/examples/config/data_config_finetune_fil.json
.OmniVoice/examples/config/ds_config_zero2_fp16.json
```

Start from:

```text
.OmniVoice/examples/config/train_config_finetune_sdpa.json
.OmniVoice/examples/config/data_config_finetune.json
.OmniVoice/examples/config/ds_config_zero2.json
```

Recommended T4-friendly settings:

- use SDPA attention instead of flex attention;
- use FP16 instead of BF16;
- lower batch size or token budget;
- start with fewer steps, such as 50 for smoke testing, then 1000-5000 for the real run;
- use gradient accumulation if needed;
- save checkpoints regularly.

### 6. Run a Smoke Test

Before full training, run a minimal test.

The smoke test should confirm:

- OmniVoice imports correctly;
- the base model can generate Filipino audio;
- tokenization finishes on a small subset;
- training starts successfully;
- a 50-step training run completes;
- a checkpoint is saved;
- inference works from the checkpoint.

Only proceed to full training after these pass.

### 7. Fine-Tune OmniVoice

Launch training with the Kaggle-specific configs.

The training command should follow the pattern from:

```text
.OmniVoice/examples/run_finetune.sh
```

Expected output:

```text
/kaggle/working/exp/omnivoice_fil_finetune/
```

Save the final checkpoint as a Kaggle output artifact or upload it to persistent storage.

### 8. Generate Matched Samples

Generate audio using both:

1. base OmniVoice: `k2-fsa/OmniVoice`;
2. Filipino fine-tuned OmniVoice checkpoint.

Use the same:

- Filipino test prompts;
- reference speaker audio;
- generation parameters;
- output naming scheme.

Recommended output folders:

```text
/kaggle/working/results/base_omnivoice_fil/
/kaggle/working/results/finetuned_omnivoice_fil/
```

### 9. Evaluate Outputs

Evaluate the base and fine-tuned outputs using the same test set.

Metrics:

- `CER/WER`: intelligibility. Use ASR to transcribe generated speech, then compare the transcript with the target text. Lower is better.
- Speaker similarity: compare speaker embeddings between reference audio and generated audio. Higher is better.
- `UTMOS`: predicted speech naturalness, if feasible. Higher is better.
- Human ratings: Filipino listeners rate pronunciation, naturalness, intelligibility, speaker similarity, and preference.

The report should compare the size of the change from base to fine-tuned model, not only whether the fine-tuned model improved.

Final full test-set objective results (controlled comparison; all fine-tunes use the same 5000-step run with the best-development-loss checkpoint, differing only in learning rate):

| Model | Selection | WER (%) | WER delta vs base | Speaker similarity (SIM-o) | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | 5000-step run, best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | 5000-step run, best development-loss checkpoint | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | 5000-step run, best development-loss checkpoint | 21.96 | -0.59 | 0.605 | 3.60 |

Strongest overall model: the best-eval LR `1e-5` checkpoint (`omnivoice-filipino-full-checkpoint-4900`). It has the lowest WER and therefore the strongest intelligibility result, with SIM-o slightly above base. LR `2e-5` is close in WER but drops SIM-o below base; LR `5e-6` keeps the best fine-tune SIM-o but gains little WER. The base model keeps the highest UTMOS, so the comparison should be reported as an intelligibility-versus-speaker-similarity/naturalness tradeoff, not a single-metric win.

### 10. Package Final Outputs

Expected research artifacts:

- standalone cleaned `data/pld_filipino_clean/` Kaggle dataset;
- dataset preparation script;
- Kaggle notebook or Kaggle script;
- Kaggle-specific OmniVoice config files;
- fine-tuned checkpoint;
- generated audio samples from base and fine-tuned models;
- objective evaluation table;
- human evaluation summary;
- final discussion of what changed and what did not.

## Next Files to Create

The next implementation phase should create:

```text
kaggle/omnivoice_filipino_finetune.ipynb
kaggle/README.md
kaggle/config/train_config_finetune_kaggle_t4.json
kaggle/config/data_config_finetune_fil.json
kaggle/config/ds_config_zero2_fp16.json
```

If notebooks are inconvenient to version, create a Kaggle-ready Python script first:

```text
kaggle/omnivoice_filipino_finetune.py
```

## Known Risks

- Kaggle T4 memory may be too limited for the default OmniVoice fine-tuning config.
- Some OmniVoice dependencies may conflict with the Kaggle base environment.
- Tokenization or training may uncover additional decode or duration edge cases.
- Automatic ASR evaluation for Filipino may be imperfect.
- UTMOS and speaker-similarity evaluation may add dependency or runtime issues.
- Fine-tuning may improve some metrics but hurt others, especially speaker similarity or naturalness.

## Definition of Done

The implementation phase is considered successful when:

1. `data/pld_filipino_clean/train.jsonl`, `data/pld_filipino_clean/dev.jsonl`, and `data/pld_filipino_clean/test.jsonl` are created and validated. Done.
2. `data/pld_filipino_clean/wavs/` contains only selected WAV files used by those manifests. Done.
3. OmniVoice tokenization produces train/dev `data.lst` files in Kaggle.
4. A 50-step Kaggle smoke test completes.
5. A real fine-tuning run completes and saves a checkpoint.
6. Base and fine-tuned models generate matched Filipino samples.
7. At least one objective metric is computed.
8. A small Filipino listener evaluation is completed or prepared.
9. Results are summarized as base vs fine-tuned metric changes.
