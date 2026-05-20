# Data

This folder contains the project-ready datasets used by the Kaggle notebooks.
These are cleaned or tokenized project artifacts, not the original raw PLD
source extraction.

## Contents

- `pld_filipino_clean/` is the cleaned Filipino PLD dataset uploaded or attached
  in Kaggle. It contains `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `wavs/`.
- `pld_filipino_clean.zip` is the zipped Kaggle upload package for the cleaned
  Filipino PLD dataset.
- `omnivoice_pld_audio_tokens_kaggle_safe/` is the tokenized OmniVoice
  audio-token dataset uploaded or attached in Kaggle. Its audio shards use the
  `.wds` extension so Kaggle does not expand WebDataset tar shards.
- `omnivoice_pld_audio_tokens_kaggle_safe.zip` is the zipped upload package for
  the Kaggle-safe tokenized dataset.

The original extracted PLD WAV/LOG tree is kept in
`../data_artifacts/original_pld/`.

