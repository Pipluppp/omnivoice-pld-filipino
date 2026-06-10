# 2026-06-11 — Showcase Web App (audio comparison)

Resume point for the OmniVoice listening-test showcase. The app is **built and
verified locally; not yet deployed, not yet committed** (as of this date), and
still missing the actual model-generated audio.

## What it is

A single-page listening-test app in `showcase/` comparing voice-clone outputs
of the base OmniVoice checkpoint against the three fine-tuned learning-rate
variants on 5 PLD Filipino test utterances. Stack:

- Vite + React + TypeScript + Tailwind v4
- shadcn/ui, monochrome **rhea** theme (scaffolded with
  `npx shadcn@latest create --template vite --preset b27GcrRo`)
- Cloudflare Workers assets-only deploy (`wrangler.jsonc`, no server code)

## Key behavior (already working)

- Sidebar lists the 5 test utterances with transcripts (taken from
  `data/pld_filipino_clean/*.jsonl`) and speaker IDs.
- Per sample: transcript, reference-recording player, "Model outputs" panel
  with a Base / LR 2e-5 / LR 5e-6 / LR 1e-5 toggle.
- **Position-preserving model switch**: swapping models mid-playback carries
  over playback time + play state (see `showcase/src/components/audio-player.tsx`,
  `preservePosition` prop). Keys `1`–`4` switch models, `d` toggles dark mode.
- **Auto-detection of audio files**: `showcase/src/hooks/use-audio-availability.ts`
  HEAD-probes each expected wav URL and checks content-type (because both Vite
  dev and Workers SPA fallback answer missing paths with index.html). Toggles
  enable themselves the moment a file exists — no code change needed when
  adding model outputs.
- Evaluation summary table with WER / SIM-o / UTMOS per model, numbers taken
  from `hyperparameter_tuning.md` (full PLD Filipino test split).

## File map

| Path | Purpose |
| --- | --- |
| `showcase/src/data/samples.ts` | The 5 samples + transcripts, model list, metrics, URL helpers |
| `showcase/src/App.tsx` | Whole UI (sidebar, players, model toggle, eval table) |
| `showcase/src/components/audio-player.tsx` | Custom player (play/seek/time, position carry-over) |
| `showcase/src/hooks/use-audio-availability.ts` | HEAD-probe availability detection |
| `showcase/public/audio/reference/` | The 5 reference wavs (already copied in) |
| `showcase/public/audio/{base,finetune_lr_2e-5,finetune_lr_5e-6,finetune_lr_1e-5}/` | Empty (`.gitkeep`), awaiting model outputs |
| `showcase/wrangler.jsonc` | Workers assets-only config |
| `showcase/README.md` | Dev/deploy/audio-naming instructions |

## The 5 chosen test utterances

```
0105.111124.050515.0394   spk 0105
0085.110923.071602.0437   spk 0085
0093.111003.081411.0343   spk 0093
0153.120215.053407.0255   spk 0153
0166.120314.005502.0171   spk 0166
```

## Commands (run inside `showcase/`)

```sh
npm run dev          # Vite dev server
npm run preview:cf   # build + wrangler dev (production build locally)
npm run deploy       # build + wrangler deploy (needs `npx wrangler login` once)
```

## To-do later

1. **Generate and add model audio** — run inference for each of the 4 models
   on the 5 utterances, save as
   `showcase/public/audio/<model-dir>/<utterance-id>.wav`
   (exact same filename as the reference wav). They light up automatically.
2. **Commit the showcase** — everything is currently uncommitted on `main`.
3. **Deploy** — `npx wrangler login`, then `npm run deploy` from `showcase/`.
4. (Optional) Bump `compatibility_date` in `wrangler.jsonc` — pinned to
   `2026-05-01` because wrangler 4.86's local runtime rejected newer dates;
   upgrading wrangler lifts this.
5. (Optional) If the controlled 5000-step LR reruns from
   `hyperparameter_tuning.md` land, update labels/metrics in
   `showcase/src/data/samples.ts` (models are described there as
   "1,000 steps", "2,000 steps", "best dev loss at step 4,900").

## Verification already done (no need to redo)

Tested via `wrangler dev` + Chrome DevTools: reference playback works, the
availability probe was exercised with temporary stand-in wavs (since removed),
mid-playback model switching carried position correctly, light/dark themes
render, console clean, `npm run build` passes type-check.
