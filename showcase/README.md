# OmniVoice Filipino Fine-tuning Showcase

A small listening-test web app comparing voice-clone outputs from the base
OmniVoice checkpoint and the three controlled fine-tunes (best
development-loss checkpoints of identical 5,000-step runs at LR 1e-5, 2e-5,
and 5e-6) on twelve PLD Filipino test utterances (5 sentences from different
speakers + 7 single words from speaker 0002).

Built with Vite + React + shadcn/ui (monochrome rhea theme), deployed as a
Cloudflare Workers assets-only project.

## Develop

```sh
npm install
npm run dev          # Vite dev server
npm run preview:cf   # build + wrangler dev (serves the production build)
```

Press `d` in the app to toggle dark mode. Key `1` selects the ground truth and
`2`–`5` the model outputs while listening; playback position carries over so
renditions can be compared at the same point in the utterance.

## Audio files

All audio is in place, exported from the WandB evaluation tables of the
Kaggle training runs. Files are named after the dataset utterance id:

```
public/audio/
  prompt/               # voice prompts — the cloning input the models heard
  reference/            # ground-truth dataset audio of the target lines
  base/                 # base OmniVoice outputs
  finetune_lr_2e-5/     # fine-tuned, 5,000-step run, best dev-loss checkpoint
  finetune_lr_5e-6/     # fine-tuned, 5,000-step run, best dev-loss checkpoint
  finetune_lr_1e-5/     # fine-tuned, 5,000-step run, best dev-loss checkpoint (step 4,900)
```

Each sample's voice prompt is a *different* utterance from the same speaker
(e.g. `…0396.wav` is the prompt for target `…0394`); the mapping lives in
`src/data/samples.ts` together with transcripts, model labels, and
evaluation metrics.

Swapping or adding generated wavs needs no code change — the app probes each
file at load time and enables a model's toggle for a sample as soon as its
wav exists. Missing files show as disabled toggles.

## Deploy to Cloudflare Workers

```sh
npx wrangler login   # once
npm run deploy       # builds and deploys per wrangler.jsonc
```
