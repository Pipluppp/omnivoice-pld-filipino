# Scripts

This folder contains local helper scripts for preparing datasets before Kaggle
runs.

- `prepare_omnivoice_filipino.py` prepares the raw Filipino PLD WAV/LOG tree
  from `data_artifacts/original_pld/` into the cleaned Kaggle-ready
  `data/pld_filipino_clean/` package.
- `repackage_omnivoice_tokens_for_kaggle.py` repackages the original tokenized
  OmniVoice WebDataset output from
  `data_artifacts/omnivoice_filipino_pld_tokens_full_original_tar/` into the
  Kaggle-safe `data/omnivoice_pld_audio_tokens_kaggle_safe/` package.

The Kaggle notebook-style training and evaluation files are in `../notebooks/`.

