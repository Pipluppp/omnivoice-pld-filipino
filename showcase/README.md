# OmniVoice Filipino Fine-tuning Showcase

A small listening-test web app comparing voice-clone outputs from the base
OmniVoice checkpoint and the three fine-tuned learning-rate variants
(LR 2e-5, LR 5e-6, LR 1e-5) on five PLD Filipino test utterances.

Built with Vite + React + shadcn/ui (monochrome rhea theme), deployed as a
Cloudflare Workers assets-only project.

## Develop

```sh
npm install
npm run dev          # Vite dev server
npm run preview:cf   # build + wrangler dev (serves the production build)
```

Press `d` in the app to toggle dark mode. Keys `1`–`4` switch between model
outputs while listening; playback position carries over so renditions can be
compared at the same point in the utterance.

## Adding model audio

The five reference recordings are already in `public/audio/reference/`.
Generated samples go next to them, named after the utterance id:

```
public/audio/
  reference/            # ground-truth dataset audio (already included)
  base/                 # base OmniVoice outputs
  finetune_lr_2e-5/     # fine-tuned, 1,000 steps
  finetune_lr_5e-6/     # fine-tuned, 2,000 steps
  finetune_lr_1e-5/     # fine-tuned, best dev loss at step 4,900
```

For example: `public/audio/base/0105.111124.050515.0394.wav`.

No code changes are needed — the app probes each file at load time and
enables a model's toggle for a sample as soon as its wav exists. Missing
files show as disabled toggles with an "Audio not added yet" tooltip.

Transcripts, model labels, and evaluation metrics live in
`src/data/samples.ts` if they ever need updating.

## Deploy to Cloudflare Workers

```sh
npx wrangler login   # once
npm run deploy       # builds and deploys per wrangler.jsonc
```
