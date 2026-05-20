# Current Trends project on Voice cloning

Working towards a course on current computing trends we need to create a research project like a mini thesis tackling a topic.

The topic of interest is voice AI, voice cloning, given that we already have a working prototype web application similar to Eleven Labs that we can reuse, where it has voice cloning and voice generation features. Workflows such as: user gives reference audio, submit to qwen ai voice model, it creates a model which can then be used for voice generation given text. We do not necessarily need to 100% stick to this application, but rather re-use it for the research project we will do.

This led to thinking of tackling voice AI topics. But then the question arises: What will be the research problem, research question, goals, methodology, what are we trying to achieve, what is the output?

**Criteria**

In terms of the research project's critiquing, the professor is already happy with the web application, but we still need the **research meat**. The problem to tackle, what we will do (methodology), what is the emerging technology component (voice AI cloning), etc.

**Scope**

The research project should be able to be accomplished in 1-2 weeks, and so the scope and scale of the project should not be overblown.

**Initial idea**

Given this we have thought of doing a fine tuning of **OmniVoice** which is a voice cloning model with multilingual capabilities, where it also has the goal language of **Filipino**. We are thinking of finetuning this model for Filipino given some data then just benchmark how well it performs to others (base non-finetuned, etc.)

- [OmniVoice paper](<references/Zhu et al - OmniVoice.pdf>) and github link https://github.com/k2-fsa/OmniVoice and some relevant finetuning Github Issue discussions https://github.com/k2-fsa/OmniVoice/issues/147
- Somehow relevant works (if it can be used for dataset for the finetuning) https://sigul-2024.ilc.cnr.it/wp-content/uploads/2024/05/Cajote-et-al.pdf https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk 

## Current research plan

Detailed execution plan: [finetuning_plan.md](finetuning_plan.md)

Presentation materials:

- [Slide content markdown](documents/current_trends/presentation/proposal_presentation.md)
- [Beamer slide source](documents/current_trends/presentation/proposal_beamer.tex)
- [Compiled slide deck](documents/current_trends/presentation/proposal_beamer.pdf)

Written proposal:

- [Proposal LaTeX source](documents/current_trends/paper/research_proposal.tex)
- [Compiled proposal PDF](documents/current_trends/paper/research_proposal.pdf)

### Working title

Low-resource Filipino adaptation of OmniVoice for multilingual zero-shot voice cloning and text-to-speech.

### Final framing

The project should not be framed as adding Filipino support to OmniVoice. OmniVoice already supports Filipino (`fil`) and lists about 7.71 hours of Filipino in its multilingual training data. The more accurate research problem is:

> How much does Filipino-specific fine-tuning improve OmniVoice performance for Filipino TTS and voice cloning across intelligibility, naturalness, and speaker similarity metrics?

This framing gives the project enough research substance while keeping the implementation feasible for a 1-2 week course project.

### Research question

Can a pretrained massively multilingual zero-shot TTS model be adapted to Filipino using a small, curated Filipino speech corpus, and does that adaptation improve generated Filipino speech over the base model?

### Objectives

1. Prepare a clean Filipino speech-text dataset suitable for OmniVoice fine-tuning.
2. Fine-tune the pretrained `k2-fsa/OmniVoice` checkpoint on Filipino read speech.
3. Compare the base and fine-tuned models using objective and subjective evaluation.
4. Reuse the existing voice cloning web application as a demo interface for generated outputs.

### Model choice

Primary model: OmniVoice

- Paper: `references/Zhu et al - OmniVoice.pdf`
- Repository: https://github.com/k2-fsa/OmniVoice
- Main reason for selection: open-source multilingual zero-shot TTS model with official fine-tuning scripts.
- Architecture summary: OmniVoice is a single-stage non-autoregressive model that directly maps text and optional voice prompts to multi-codebook acoustic tokens using a diffusion language model-style masked-token prediction objective.

The cloned repo is located at:

```text
.OmniVoice/
```

Important repo files:

- `examples/run_finetune.sh` - main fine-tuning pipeline.
- `examples/config/train_config_finetune_sdpa.json` - safer starting config for Kaggle T4 GPUs.
- `examples/config/data_config_finetune.json` - points training to tokenized WebDataset manifests.
- `docs/data_preparation.md` - describes JSONL and WebDataset conversion.
- `docs/evaluation.md` - describes supported WER, speaker similarity, and UTMOS evaluation.

### Dataset decision

Primary dataset: UP-DSP Philippine Languages Database (UP-DSP-PLD)

- Paper: https://aclanthology.org/2024.sigul-1.32.pdf
- Released dataset: https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk
- License: CC-BY-NC-4.0
- Format: WAV audio plus LOG transcript files
- Size: 45.63 GB
- Intended uses include ASR, phoneme transcription, voice conversion, and TTS.

The dataset fits OmniVoice fine-tuning because OmniVoice only needs audio paths, transcripts, and optional language IDs in JSONL format:

```jsonl
{"id":"fil_000001","audio_path":"/path/to/audio.wav","text":"Magandang umaga sa inyong lahat.","language_id":"fil"}
```

Relevant Filipino subset from the PLD paper:

- Language ID: `fil`
- Speakers: 135
- Utterances: 52,879
- Duration: 48:56:36
- Average utterance length: 3.5796 seconds

This is enough for a small fine-tuning study. For the first experiment, use only the Filipino read-speech subset. Avoid spontaneous speech initially because read-speech transcripts are already matched to prompts, while spontaneous speech may require more transcript quality checking.

### Current prepared dataset

The local pre-Kaggle dataset preparation is now complete. The upload-ready dataset is:

```text
data/pld_filipino_clean/
  wavs/
  train.jsonl
  dev.jsonl
  test.jsonl
```

This directory is intentionally standalone. It contains only the selected WAV files and the OmniVoice JSONL manifests needed for training/evaluation upload, without audit metadata or backup files.

Current split:

| Split | Samples | Percent | Hours | Speakers |
|---|---:|---:|---:|---:|
| train | 33,448 | 79.12% | 31.902 | 103 |
| dev | 4,506 | 10.66% | 3.705 | 13 |
| test | 4,322 | 10.22% | 4.165 | 13 |
| total | 42,276 | 100% | 39.772 | 129 |

Applied filters:

- kept read/prompted Filipino speech only;
- excluded spontaneous prompt sources;
- excluded rows whose transcript contains parentheses;
- excluded rows whose transcript contains digits;
- excluded unreadable or zero-byte WAV files;
- kept clips from 1.0 to 15.0 seconds;
- split speakers disjointly across train/dev/test.

### Dataset constraints and ethics

The PLD dataset is suitable for academic research, but it has important constraints:

- It is non-commercial because of CC-BY-NC-4.0.
- Mozilla Data Collective notes that redistribution, third-party sharing, or commercial use requires prior written consent.
- The dataset should not be used to identify speakers.
- The resulting project should be presented as a research prototype, not a commercial Filipino voice cloning product.

### Data preparation plan

1. Download the PLD dataset from Mozilla Data Collective. Done.
2. Extract the Filipino (`fil`) subset locally under `data/`. Done.
3. Parse LOG files to map each WAV file to its transcript. Done.
4. Filter to clean read speech, excluding spontaneous, parenthetical, digit-containing, unreadable, zero-byte, too-short, and too-long samples. Done.
5. Create speaker-disjoint train/dev/test splits. Done.
6. Convert each split into OmniVoice JSONL format under `data/pld_filipino_clean/`. Done.

Regenerate the clean package with:

```powershell
python scripts\prepare_omnivoice_filipino.py --package-dir data\pld_filipino_clean --package-mode hardlink
```

### Kaggle T4 fine-tuning plan

Target environment: Kaggle notebook with 2x NVIDIA T4 GPUs.

The default OmniVoice fine-tuning config uses BF16 and a large token budget. For T4 GPUs, use a more conservative setup:

- Use `fp16`, not `bf16`.
- Use SDPA attention, not the default flex attention.
- Use smaller `batch_tokens`.
- Use DeepSpeed ZeRO-2 if memory is tight.
- Start with a smaller experiment before training on all Filipino data.

Recommended first training experiment:

- Data: start with a small smoke-test subset, then train on the full 39.772-hour `data/pld_filipino_clean/` package if tokenization and the smoke test pass.
- Steps: 1,000-2,000 first, then increase if loss and samples improve.
- Learning rate: `5e-6` to `1e-5`.
- Batch tokens: start at `1024`; reduce to `768` or `512` if out of memory.
- Gradient accumulation: `4`.
- Max sample tokens: `900-1200`.
- Save checkpoints every 250-500 steps.

Suggested Kaggle-oriented config changes:

```json
{
  "init_from_checkpoint": "k2-fsa/OmniVoice",
  "learning_rate": 5e-6,
  "steps": 2000,
  "batch_tokens": 1024,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "fp16",
  "allow_tf32": false,
  "attn_implementation": "sdpa",
  "max_sample_tokens": 1200,
  "min_sample_tokens": 50,
  "max_batch_size": 16,
  "use_deepspeed": true,
  "deepspeed_config": "config/ds_config_zero2_fp16.json",
  "logging_steps": 25,
  "eval_steps": 250,
  "save_steps": 250,
  "keep_last_n_checkpoints": 2
}
```

If training runs out of memory, reduce in this order:

1. `batch_tokens`: `1024` to `768` to `512`
2. `max_batch_size`: `16` to `8`
3. `max_sample_tokens`: `1200` to `900`
4. disable dev evaluation during early smoke tests

### Fine-tuning workflow

1. Install OmniVoice and dependencies in Kaggle.
2. Upload or attach the local `data/pld_filipino_clean/` directory as a Kaggle dataset.
3. Copy the mounted Kaggle dataset into `/kaggle/working/data_clean/` if the input mount is read-only.
4. Tokenize `train.jsonl` and `dev.jsonl` using `omnivoice.scripts.extract_audio_tokens`.
5. Update `examples/config/data_config_finetune.json` to point to the generated `data.lst` files.
6. Launch training with `accelerate`.
7. Save the final checkpoint to Kaggle output or external storage.
8. Use both the base and fine-tuned checkpoints to generate the same Filipino test sentences.

### Evaluation plan and current objective results

Compare at least two systems:

1. Base OmniVoice: `k2-fsa/OmniVoice`
2. Filipino fine-tuned OmniVoice

Additional fine-tuned checkpoints are compared when available:

3. 1000-step full Filipino PLD checkpoint, learning rate `2e-5`.
4. 2000-step full Filipino PLD checkpoint, learning rate `5e-6`.
5. Best development-loss checkpoint from a 5000-step run, learning rate `1e-5`.

Metrics:

- CER or WER: use Whisper-large-v3, Whisper-large-v3-turbo, or another Filipino-capable ASR model to transcribe generated speech and compare against target text. Lower error means the generated audio is more intelligible.
- Speaker similarity: use ECAPA-TDNN or WavLM-based speaker embeddings when reference audio is used. Higher similarity means the cloned voice is closer to the reference speaker.
- Naturalness: use UTMOS if the setup is manageable. Higher score means the audio sounds more natural according to the predictor.
- Human evaluation: ask Filipino speakers to rate pronunciation, naturalness, intelligibility, speaker similarity, and preference on a small sample set.

Current full test-set objective metrics:

| Model | Checkpoint / artifact | Learning rate | Selection | WER (%) | SIM-o | UTMOS |
|---|---|---:|---|---:|---:|---:|
| Base OmniVoice | `k2-fsa/OmniVoice` | N/A | pretrained base | 22.55 | 0.602 | 3.64 |
| Fine-tuned 1000 | `omnivoice-filipino-full-checkpoint-1000` | `2e-5` | 1000 steps | 20.07 | 0.610 | 3.60 |
| Fine-tuned 2000 | `omnivoice-filipino-full-checkpoint-2000` | `5e-6` | 2000 steps | 22.64 | 0.611 | 3.57 |
| Fine-tuned best | `omnivoice-filipino-full-checkpoint-best` | `1e-5` | best eval loss at step 4900 | 18.52 | 0.604 | 3.61 |

Current best model: `omnivoice-filipino-full-checkpoint-best`.

This checkpoint is the best overall because it has the lowest WER, improving from 22.55% to 18.52% compared with the base model. That is a 4.03 absolute point reduction, or about 17.9% relative WER reduction. It does not have the highest SIM-o or UTMOS, but its SIM-o remains above the base model and its UTMOS drop is small. The 2000-step `5e-6` checkpoint has the highest SIM-o but is not the best model because its WER is slightly worse than the base model.

### Expected output

The project output should be:

1. A short research paper or mini-thesis report.
2. A reproducible fine-tuning notebook or script for Kaggle.
3. The standalone cleaned `data/pld_filipino_clean/` dataset package.
4. Generated audio comparisons from base vs fine-tuned OmniVoice.
5. Objective and subjective evaluation tables.
6. A demo integration in the existing voice cloning web app, if time allows.

### Risks

- Kaggle T4 memory may be insufficient for the default fine-tuning config.
- Full PLD processing may exceed notebook time or storage limits.
- Noisy or mismatched transcripts can make fine-tuning worse.
- Fine-tuning on Filipino only may slightly reduce performance on other languages.
- The base OmniVoice model may already perform well on Filipino, so improvements may be small.

### Success criteria

The project is successful if it can show one of the following:

1. Fine-tuning produces measurable changes in Filipino CER/WER, naturalness, or speaker similarity compared with the base model.
2. Fine-tuning does not improve results, but the project identifies why, such as data quality, base-model strength, or compute constraints.
3. A small curated subset is enough to produce measurable adaptation behavior compared with base OmniVoice.

Even a negative result is acceptable if the methodology is clear and the evaluation is defensible.
