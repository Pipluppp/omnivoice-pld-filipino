# Showcase — Agent Notes

Listening-test web app comparing voice-clone outputs of the base OmniVoice
checkpoint vs. the three controlled fine-tunes — best development-loss
checkpoints of identical 5,000-step runs at LR 2e-5, LR 5e-6, and LR 1e-5 —
on twelve PLD Filipino test utterances (5 sentences from different
speakers + 7 single words from speaker 0002). Part of the parent research
project (see `../moc.md`, `../progress/2026-06-11-showcase-web-app.md`, and
`../progress/2026-06-11-controlled-new-lr-eval.md`).

## Stack

- Vite + React 19 + TypeScript + Tailwind v4
- shadcn/ui with the monochrome **rhea** theme (`radix-rhea` style, preset
  `b27GcrRo`). All theme colors are zero-chroma oklch — keep the design
  monochrome; do not introduce colored accents.
- Cloudflare Workers **assets-only** deploy (`wrangler.jsonc`, no server code,
  no main entry).

## Commands (run in this directory)

```sh
npm run dev          # Vite dev server (hot reload)
npm run build        # tsc -b + vite build -> dist/
npm run preview:cf   # build + wrangler dev (production build, local Workers runtime)
npm run deploy       # build + wrangler deploy (needs `npx wrangler login` once)
npm run lint         # eslint
npx shadcn@latest add <component>   # add more shadcn components
```

In-app keys: `d` toggles dark mode (handled by `src/components/theme-provider.tsx`),
`1` selects ground truth and `2`–`5` the model outputs while listening.

## File map

| Path | Purpose |
| --- | --- |
| `src/data/samples.ts` | Single source of truth: the 12 samples + transcripts + prompt mapping, the 4 models (labels, audio dirs, WER/SIM-o/UTMOS metrics), URL helpers |
| `src/App.tsx` | Entire UI: how-it-works steps, sidebar, target text, voice-prompt card, compare panel (ground truth + models in one toggle), eval table |
| `src/components/audio-player.tsx` | Custom `<audio>` player. `preservePosition` carries playback time + play state across `src` changes (the core A/B-comparison feature — don't break it) |
| `src/hooks/use-audio-availability.ts` | HEAD-probes each expected wav and checks content-type to detect which files exist |
| `public/audio/prompt/` | The 12 voice prompts — the cloning input the models heard (a different utterance from the same speaker) |
| `public/audio/reference/` | The 12 ground-truth wavs of the target lines |
| `public/audio/{base,finetune_lr_2e-5,finetune_lr_5e-6,finetune_lr_1e-5}/` | Generated model outputs (12 wavs each) |
| `wrangler.jsonc` | Workers config (assets-only, SPA fallback) |

## Audio file convention

All wavs are named exactly after the dataset utterance id, e.g.
`public/audio/base/0105.111124.050515.0394.wav`. Prompt files use the
*prompt's own* utterance id (e.g. `…0396.wav` is the prompt for target
`…0394`); the target→prompt mapping is `promptId` in `src/data/samples.ts`.
The generated wavs were exported from the WandB evaluation tables of the
Kaggle runs:

- `base` ← `evaluation_audio_examples` table of the first full eval run
  (local raw export `../step-1000/`).
- `finetune_lr_1e-5` ← `finetune_evaluation_audio_examples` table,
  `finetuned_best` rows = step-4900 best-dev-loss checkpoint of the 5000-step
  LR 1e-5 run (local raw export `../step-2000-and-best-eval-4900/`).
- `finetune_lr_2e-5` / `finetune_lr_5e-6` ← `finetune_evaluation_audio_examples`
  table of the controlled new-LR eval run
  (`../eval-2e-5-and-5e-6/run-omnivoice-fil-new-lr-eval-892960a1-…`),
  `best_eval_lr_2e_5` / `best_eval_lr_5e_6` rows = best-dev-loss checkpoints
  of the controlled 5000-step runs.

No code change is needed when swapping files: the availability hook detects
them at load time and enables the corresponding model toggle.

## Gotchas

- **Missing-file detection**: both the Vite dev server and Workers SPA
  fallback answer missing paths with `index.html` (HTTP 200), so availability
  is determined by content-type, not status code. Keep that check if you touch
  the hook.
- **`compatibility_date` is pinned to 2026-05-01** because the local runtime
  bundled with wrangler 4.86 rejects newer dates. Upgrade wrangler before
  raising it.
- The evaluation metrics in `samples.ts` are the final controlled-comparison
  values from `../hyperparameter_tuning.md` and
  `../eval-2e-5-and-5e-6/metrics/metric_summary.json` (full PLD Filipino test
  split). Update them together if results change.
- `.wrangler/` is local dev state and gitignored; never commit it.

## Verifying changes

`npm run preview:cf`, then open http://localhost:8787. Check: reference player
plays; model toggles are disabled while their wavs are absent (drop a test wav
into `dist/audio/<model>/` to light one up without touching `public/`);
switching models mid-playback keeps the playback position; light + dark themes
both render; console is clean.
