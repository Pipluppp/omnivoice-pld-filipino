# 2026-05-18 OmniVoice Kaggle Smoke Progress

## Accomplished

https://www.kaggle.com/code/pipluppp/omnivoice-pld-vibe-checks-that-it-s-running/output

- Inspected OmniVoice fine-tuning flow and confirmed we can reuse the repo scripts.
- Wrote `omnivoice_kaggle_initial.py` for Kaggle Script-to-Notebook use.
- Fixed Kaggle path wiring:
  - OmniVoice repo: `/kaggle/working/OmniVoice`
  - PLD dataset: `/kaggle/input/datasets/pipluppp/pld-filipino-cleaned/data_clean`
- Switched setup to `uv sync` and `uv run`.
- Rewrote PLD manifest audio paths to absolute Kaggle paths.
- Fixed WebDataset key mismatch by sanitizing dotted PLD IDs.
- Tokenized smoke data successfully:
  - train: 96/96 samples, 0 errors, 2 shards
  - dev: 24/24 samples, 0 errors, 1 shard
- Ran 5-step fine-tuning smoke test successfully.
- Saved checkpoint:
  `/kaggle/working/omnivoice_filipino_pld/exp_smoke/checkpoint-5`
- Ran smoke inference from the checkpoint and generated:
  `/kaggle/working/omnivoice_filipino_pld/results/filipino_smoke.wav`

## Notes

- The smoke run proves the full pipeline works, not model quality.
- Kaggle `sitecustomize` reports missing `wrapt`; harmless, but the script now installs it in the uv env.
- Trainer saves checkpoint twice at step 5 because `save_steps=5` and final save both trigger.

## Next

- Run full tokenization once with `SMOKE_TEST = False`, `RUN_TRAINING = False`.
- Preserve `tokens_full/` as Kaggle output so it does not need to be regenerated.
- Start real training with conservative settings: 500 steps, `batch_tokens` 2048 or 4096, save/eval every 100 steps.

# 2026-05-18 More work afterwards

Did a first full training run on Kaggle to exercise the pipeline end to end; it took about an hour. The audio tokenization took time, so did work on reusing the created tokens as a Kaggle dataset. Training scripts now loads this instead of doing the tokenization on the fly for each training. Still no post-training immediate evaluation of the metrics used in OmniVoice. Still gauging around the training setups and how it looks.

Next work:

- [ ] Setup the Evaluation metrics notebooks to load finetune models and base, run the metrics and the models needed to compute the metrics, run inference on test samples, check out how it performs, etc.