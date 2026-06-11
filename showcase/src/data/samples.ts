export interface Sample {
  /** Dataset utterance id, also the wav filename (without extension). */
  id: string
  /** PLD speaker directory the utterance came from. */
  speaker: string
  /** Ground-truth Filipino transcript. */
  text: string
  /**
   * Utterance id of the voice prompt: the recording the models actually
   * heard as the cloning input — a different line from the same speaker.
   */
  promptId: string
  /** Transcript of the voice prompt. */
  promptText: string
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
    detail: "Fine-tuned, 5,000-step run, best dev-loss checkpoint (step 5,000)",
    metrics: { wer: 18.83, werDelta: -3.72, simO: 0.583, utmos: 3.61 },
  },
  {
    id: "ft-5e-6",
    dir: "finetune_lr_5e-6",
    label: "LR 5e-6",
    detail: "Fine-tuned, 5,000-step run, best dev-loss checkpoint (step 5,000)",
    metrics: { wer: 21.96, werDelta: -0.59, simO: 0.605, utmos: 3.6 },
  },
  {
    id: "ft-1e-5",
    dir: "finetune_lr_1e-5",
    label: "LR 1e-5",
    detail: "Fine-tuned, 5,000-step run, best dev-loss checkpoint (step 4,900)",
    metrics: { wer: 18.52, werDelta: -4.03, simO: 0.604, utmos: 3.61 },
  },
]

export const SAMPLES: Sample[] = [
  {
    id: "0105.111124.050515.0394",
    speaker: "0105",
    text: "Pero kampante naman ang EU na makakahanap ng agarang solusyon ang Pilipinas patungkol sa naturang isyu.",
    promptId: "0105.111124.050515.0396",
    promptText:
      "Nakahanda ang bansa na kunin ang malaking parte ng pamilihang-mangga sa Estados Unidos, ayun sa sekretaryo ng Departamento ng Agrikultura.",
  },
  {
    id: "0085.110923.071602.0437",
    speaker: "0085",
    text: "Nuong panahon nang ang Pilipinas ay nasasakop pa ng mga Kastila ay mayruong mag-asawang naninirahan sa paanan ng bundok ng San Mateo, Rizal.",
    promptId: "0085.110923.071602.0438",
    promptText:
      '"Naku", ang himutok ng matsing, "matagal nang patay! At ang iyo, Binibining Pagong?"',
  },
  {
    id: "0093.111003.081411.0343",
    speaker: "0093",
    text: "Nakahanda ang bansa na kunin ang malaking parte ng pamilihang-mangga sa Estados Unidos, ayun sa sekretaryo ng Departamento ng Agrikultura.",
    promptId: "0093.111003.081411.0344",
    promptText: "Ang produksiyon ng agrikultura ay tumaas",
  },
  {
    id: "0153.120215.053407.0255",
    speaker: "0153",
    text: "Naniniwala ang mga mangangalakal na lalong lalago ang negosyo sa lungsod ng Cotabato",
    promptId: "0153.120215.053407.0257",
    promptText:
      "Ibinabala ng Asian Development Bank of the Philippines na walang tsansang umangat ang ekonomiya ng bansa sa lalong madaling panahon.",
  },
  {
    id: "0166.120314.005502.0171",
    speaker: "0166",
    text: "Kung gayon, dapat ay gawin natin ang ating tungkulin na mas ipaalam ang tunay na peligro ng pag-inom ng alak lalo na sa nagiging alipin dito.",
    promptId: "0166.120314.005502.0172",
    promptText:
      "Kung sinabing lumikas mula sa danger zone, sumunod, para sa ikabubuti ng iyong kalusugan.",
  },
  // Single-word utterances from speaker 0002 (Filipino kinship terms) —
  // short word-level cloning tests from the same WandB evaluation tables.
  {
    id: "0002.110816.100658.0003",
    speaker: "0002",
    text: "diko",
    promptId: "0002.110816.100658.0004",
    promptText: "bunso",
  },
  {
    id: "0002.110816.100658.0004",
    speaker: "0002",
    text: "bunso",
    promptId: "0002.110816.100658.0005",
    promptText: "amang",
  },
  {
    id: "0002.110816.100658.0005",
    speaker: "0002",
    text: "amang",
    promptId: "0002.110816.100658.0007",
    promptText: "katambal",
  },
  {
    id: "0002.110816.100658.0007",
    speaker: "0002",
    text: "katambal",
    promptId: "0002.110816.100658.0008",
    promptText: "katukayo",
  },
  {
    id: "0002.110816.100658.0008",
    speaker: "0002",
    text: "katukayo",
    promptId: "0002.110816.100658.0009",
    promptText: "kaibigan",
  },
  {
    id: "0002.110816.100658.0009",
    speaker: "0002",
    text: "kaibigan",
    promptId: "0002.110816.100658.0010",
    promptText: "kinakapatid",
  },
  {
    id: "0002.110816.100658.0010",
    speaker: "0002",
    text: "kinakapatid",
    promptId: "0002.110816.100658.0011",
    promptText: "kapitbahay",
  },
]

export function referenceUrl(sample: Sample) {
  return `/audio/reference/${sample.id}.wav`
}

export function promptUrl(sample: Sample) {
  return `/audio/prompt/${sample.promptId}.wav`
}

export function modelUrl(model: ModelInfo, sample: Sample) {
  return `/audio/${model.dir}/${sample.id}.wav`
}
