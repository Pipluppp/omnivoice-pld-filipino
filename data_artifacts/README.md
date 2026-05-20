# Data Artifacts

This folder is for source, raw, bulky, or intermediate data artifacts. It is not
the folder used directly by the Kaggle notebooks.

## Contents

- `original_pld/` is the original extracted Filipino PLD WAV/LOG tree.
- `filipino_omnivoice_manifests/` is an earlier local OmniVoice-style manifest
  output with audit files.
- `omnivoice_filipino_pld_tokens_full_original_tar/` is the original tokenized
  OmniVoice output with tar-format WebDataset shards. It contains the same
  train/dev token counts as the Kaggle-safe token package, but it was not the
  final upload form because Kaggle may expand `.tar` shards.

Kaggle-ready datasets live under `../data/`.

