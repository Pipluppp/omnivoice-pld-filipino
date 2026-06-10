export interface Sample {
  /** Dataset utterance id, also the wav filename (without extension). */
  id: string
  /** PLD speaker directory the utterance came from. */
  speaker: string
  /** Ground-truth Filipino transcript. */
  text: string
}

export interface ModelInfo {
  id: string
  /** Directory under /audio/ holding this model's generated wavs. */
  dir: string
  label: string
  detail: string
  /** Evaluated on the full PLD Filipino test split. */
  metrics: {
    wer: number
    werDelta: number
    simO: number
    utmos: number
  }
}

export const MODELS: ModelInfo[] = [
  {
    id: "base",
    dir: "base",
    label: "Base",
    detail: "Pretrained OmniVoice, zero-shot cloning",
    metrics: { wer: 22.55, werDelta: 0, simO: 0.602, utmos: 3.64 },
  },
  {
    id: "ft-2e-5",
    dir: "finetune_lr_2e-5",
    label: "LR 2e-5",
    detail: "Fine-tuned, 1,000 steps",
    metrics: { wer: 20.07, werDelta: -2.48, simO: 0.61, utmos: 3.6 },
  },
  {
    id: "ft-5e-6",
    dir: "finetune_lr_5e-6",
    label: "LR 5e-6",
    detail: "Fine-tuned, 2,000 steps",
    metrics: { wer: 22.64, werDelta: 0.09, simO: 0.611, utmos: 3.57 },
  },
  {
    id: "ft-1e-5",
    dir: "finetune_lr_1e-5",
    label: "LR 1e-5",
    detail: "Fine-tuned, best dev loss at step 4,900",
    metrics: { wer: 18.52, werDelta: -4.03, simO: 0.604, utmos: 3.61 },
  },
]

export const SAMPLES: Sample[] = [
  {
    id: "0105.111124.050515.0394",
    speaker: "0105",
    text: "Pero kampante naman ang EU na makakahanap ng agarang solusyon ang Pilipinas patungkol sa naturang isyu.",
  },
  {
    id: "0085.110923.071602.0437",
    speaker: "0085",
    text: "Nuong panahon nang ang Pilipinas ay nasasakop pa ng mga Kastila ay mayruong mag-asawang naninirahan sa paanan ng bundok ng San Mateo, Rizal.",
  },
  {
    id: "0093.111003.081411.0343",
    speaker: "0093",
    text: "Nakahanda ang bansa na kunin ang malaking parte ng pamilihang-mangga sa Estados Unidos, ayun sa sekretaryo ng Departamento ng Agrikultura.",
  },
  {
    id: "0153.120215.053407.0255",
    speaker: "0153",
    text: "Naniniwala ang mga mangangalakal na lalong lalago ang negosyo sa lungsod ng Cotabato",
  },
  {
    id: "0166.120314.005502.0171",
    speaker: "0166",
    text: "Kung gayon, dapat ay gawin natin ang ating tungkulin na mas ipaalam ang tunay na peligro ng pag-inom ng alak lalo na sa nagiging alipin dito.",
  },
]

export function referenceUrl(sample: Sample) {
  return `/audio/reference/${sample.id}.wav`
}

export function modelUrl(model: ModelInfo, sample: Sample) {
  return `/audio/${model.dir}/${sample.id}.wav`
}
