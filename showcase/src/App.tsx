import * as React from "react"
import { AudioLines } from "lucide-react"

import { AudioPlayer } from "@/components/audio-player"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Kbd } from "@/components/ui/kbd"
import { Separator } from "@/components/ui/separator"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  MODELS,
  SAMPLES,
  modelUrl,
  promptUrl,
  referenceUrl,
  type ModelInfo,
  type Sample,
} from "@/data/samples"
import { useAudioAvailability } from "@/hooks/use-audio-availability"

const ALL_MODEL_URLS = SAMPLES.flatMap((sample) =>
  MODELS.map((model) => modelUrl(model, sample))
)

const GROUND_TRUTH_ID = "truth"

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-sm tabular-nums">{value}</span>
    </div>
  )
}

function HowItWorks() {
  const steps = [
    {
      title: "Models hear a voice prompt",
      detail:
        "a short clip of the speaker saying a different line — the only audio they get.",
    },
    {
      title: "Models must say the target text",
      detail:
        "in that speaker's voice, without ever hearing the real recording of it.",
    },
    {
      title: "You compare the attempts",
      detail:
        "each AI clone next to the ground truth — the real speaker reading the same line.",
    },
  ]
  return (
    <ol className="grid gap-3 rounded-lg border bg-muted/30 p-4 sm:grid-cols-3">
      {steps.map((step, i) => (
        <li key={step.title} className="flex gap-3">
          <span className="flex size-5 shrink-0 items-center justify-center rounded-full border font-mono text-[10px] text-muted-foreground">
            {i + 1}
          </span>
          <p className="text-xs leading-relaxed text-muted-foreground">
            <span className="font-medium text-foreground">{step.title}</span>{" "}
            — {step.detail}
          </p>
        </li>
      ))}
    </ol>
  )
}

function ComparePanel({
  sample,
  availability,
}: {
  sample: Sample
  availability: Record<string, boolean>
}) {
  const [selectedId, setSelectedId] = React.useState(GROUND_TRUTH_ID)

  const isAvailable = (model: ModelInfo) =>
    availability[modelUrl(model, sample)] === true
  const availableModel = MODELS.find((m) => m.id === selectedId && isAvailable(m))
  const selectedModel = selectedId === GROUND_TRUTH_ID ? null : availableModel
  // Fall back to ground truth if the chosen model has no audio for this sample.
  const effectiveId = selectedModel ? selectedModel.id : GROUND_TRUTH_ID
  const src = selectedModel ? modelUrl(selectedModel, sample) : referenceUrl(sample)

  // Number keys: 1 = ground truth, 2-5 = model outputs.
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (event.target instanceof HTMLElement && event.target.closest("input, textarea")) return
      if (event.key === "1") {
        setSelectedId(GROUND_TRUTH_ID)
        return
      }
      const model = MODELS[Number.parseInt(event.key, 10) - 2]
      if (model && availability[modelUrl(model, sample)]) {
        setSelectedId(model.id)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [availability, sample])

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle>Listen &amp; compare</CardTitle>
          <Badge variant="outline">step 3 · output</Badge>
        </div>
        <CardDescription className="text-pretty">
          Ground truth is the real recording; the rest are AI clones. Switch
          while playing — the position is kept, so you hear the exact same
          moment in a different voice.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <ToggleGroup
          type="single"
          variant="outline"
          spacing={0}
          value={effectiveId}
          onValueChange={(value) => value && setSelectedId(value)}
          className="w-full max-w-full overflow-x-auto"
        >
          <ToggleGroupItem value={GROUND_TRUTH_ID} className="h-10 flex-1">
            <span>Ground truth</span>
            <Kbd className="ml-1.5 hidden sm:inline-flex">1</Kbd>
          </ToggleGroupItem>
          {MODELS.map((model, index) => {
            const available = isAvailable(model)
            const item = (
              <ToggleGroupItem
                key={model.id}
                value={model.id}
                disabled={!available}
                className="h-10 flex-1"
              >
                <span>{model.label}</span>
                <Kbd className="ml-1.5 hidden sm:inline-flex">{index + 2}</Kbd>
              </ToggleGroupItem>
            )
            if (available) return item
            return (
              <Tooltip key={model.id}>
                <TooltipTrigger asChild>
                  <span className="flex flex-1">{item}</span>
                </TooltipTrigger>
                <TooltipContent>Audio not added yet</TooltipContent>
              </Tooltip>
            )
          })}
        </ToggleGroup>

        {/* Keyed per sample: the position carry-over is for model switches
            within an utterance, not across different utterances. */}
        <AudioPlayer key={sample.id} src={src} preservePosition />
        <Separator />
        {selectedModel ? (
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <span className="text-xs text-muted-foreground">
              AI-generated · {selectedModel.detail}
            </span>
            <div className="flex flex-wrap gap-x-6 gap-y-2">
              <MetricChip
                label="WER"
                value={`${selectedModel.metrics.wer.toFixed(2)}%`}
              />
              <MetricChip
                label="SIM-o"
                value={selectedModel.metrics.simO.toFixed(3)}
              />
              <MetricChip
                label="UTMOS"
                value={selectedModel.metrics.utmos.toFixed(2)}
              />
            </div>
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">
            The real speaker's recording — the models never heard this clip.
          </span>
        )}
      </CardContent>
    </Card>
  )
}

function EvalTable() {
  const bestWer = Math.min(...MODELS.map((m) => m.metrics.wer))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Evaluation summary</CardTitle>
        <CardDescription className="text-pretty">
          Scored on the full PLD Filipino test split, not just the clips on
          this page. Lower WER (transcription errors) is better; higher SIM-o
          (voice similarity) and UTMOS (naturalness) are better.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="py-2 pr-4 font-normal">Model</th>
                <th className="py-2 pr-4 font-normal">Training</th>
                <th className="py-2 pr-4 text-right font-normal">WER %</th>
                <th className="py-2 pr-4 text-right font-normal">Δ vs base</th>
                <th className="py-2 pr-4 text-right font-normal">SIM-o</th>
                <th className="py-2 text-right font-normal">UTMOS</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {MODELS.map((model) => {
                const best = model.metrics.wer === bestWer
                return (
                  <tr
                    key={model.id}
                    className={`border-b last:border-0 ${best ? "font-medium" : ""}`}
                  >
                    <td className="py-2 pr-4 font-sans">
                      {model.label}
                      {best && (
                        <Badge variant="outline" className="ml-2">
                          best WER
                        </Badge>
                      )}
                    </td>
                    <td className="py-2 pr-4 font-sans text-xs text-muted-foreground">
                      {model.detail}
                    </td>
                    <td className="py-2 pr-4 text-right">
                      {model.metrics.wer.toFixed(2)}
                    </td>
                    <td className="py-2 pr-4 text-right text-muted-foreground">
                      {model.metrics.werDelta === 0
                        ? "—"
                        : `${model.metrics.werDelta > 0 ? "+" : ""}${model.metrics.werDelta.toFixed(2)}`}
                    </td>
                    <td className="py-2 pr-4 text-right">
                      {model.metrics.simO.toFixed(3)}
                    </td>
                    <td className="py-2 text-right">
                      {model.metrics.utmos.toFixed(2)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}

export function App() {
  const [sampleId, setSampleId] = React.useState(SAMPLES[0].id)
  // Sections animate in when a sample is switched, but not on page load.
  const [hasSwitched, setHasSwitched] = React.useState(false)
  const availability = useAudioAvailability(ALL_MODEL_URLS)
  const sample = SAMPLES.find((s) => s.id === sampleId) ?? SAMPLES[0]
  const sampleIndex = SAMPLES.indexOf(sample)

  return (
    <TooltipProvider>
      <div className="min-h-svh bg-background text-foreground">
        <header className="border-b">
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-5">
            <div className="flex items-center gap-3">
              <AudioLines className="size-5" aria-hidden />
              <div>
                <h1 className="text-sm font-semibold tracking-tight">
                  OmniVoice · Filipino Fine-tuning
                </h1>
                <p className="text-xs text-muted-foreground">
                  Can an AI clone a Filipino voice from one short recording?
                  Compare the base model against three fine-tuned variants.
                </p>
              </div>
            </div>
            <p className="hidden text-xs text-muted-foreground sm:block">
              <Kbd>d</Kbd> dark mode
            </p>
          </div>
        </header>

        <main className="mx-auto grid max-w-5xl gap-8 px-6 py-8 lg:grid-cols-[260px_1fr]">
          <nav aria-label="Samples" className="min-w-0">
            <h2 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Test utterances
            </h2>
            <ul className="flex gap-2 overflow-x-auto pb-2 lg:flex-col lg:overflow-visible lg:pb-0">
              {SAMPLES.map((s, i) => {
                const active = s.id === sample.id
                return (
                  <li key={s.id} className="min-w-56 flex-1 lg:min-w-0">
                    <button
                      type="button"
                      onClick={() => {
                        setSampleId(s.id)
                        setHasSwitched(true)
                      }}
                      aria-current={active ? "true" : undefined}
                      className={`w-full rounded-lg border px-3 py-2.5 text-left transition-[color,background-color,border-color,scale] active:scale-[0.96] ${
                        active
                          ? "border-foreground/30 bg-muted"
                          : "border-border hover:bg-muted/50"
                      }`}
                    >
                      <span className="mb-1 flex items-center justify-between font-mono text-xs text-muted-foreground">
                        <span>{String(i + 1).padStart(2, "0")}</span>
                        <span>spk {s.speaker}</span>
                      </span>
                      <span className="line-clamp-2 text-xs leading-relaxed">
                        {s.text}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          </nav>

          <div className="flex min-w-0 flex-col gap-6">
            <HowItWorks />

            <section
              key={sample.id}
              className={
                hasSwitched ? "motion-safe:animate-section-in" : undefined
              }
            >
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <Badge variant="secondary">
                  Sample {String(sampleIndex + 1).padStart(2, "0")}
                </Badge>
                <Badge variant="outline">Filipino</Badge>
                <Badge variant="outline">speaker {sample.speaker}</Badge>
                <span className="font-mono text-xs text-muted-foreground">
                  {sample.id}
                </span>
              </div>
              <h2 className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Target text
              </h2>
              <blockquote className="text-pretty text-lg leading-relaxed">
                “{sample.text}”
              </blockquote>
            </section>

            <Card
              key={`prompt-${sample.id}`}
              className={
                hasSwitched
                  ? "motion-safe:animate-section-in motion-safe:[animation-delay:80ms]"
                  : undefined
              }
            >
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <CardTitle>Voice prompt</CardTitle>
                  <Badge variant="outline">steps 1–2 · input</Badge>
                </div>
                <CardDescription className="text-pretty">
                  The only audio the models heard: speaker {sample.speaker}{" "}
                  saying a <em>different</em> line.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                <AudioPlayer src={promptUrl(sample)} />
                <p className="text-xs italic text-muted-foreground">
                  “{sample.promptText}”
                </p>
              </CardContent>
            </Card>

            <ComparePanel sample={sample} availability={availability} />

            <EvalTable />
          </div>
        </main>

        <footer className="border-t">
          <div className="mx-auto max-w-5xl px-6 py-5 text-xs text-muted-foreground">
            OmniVoice Filipino fine-tuning · PLD dataset · PUP Computing Trends,
            2026
          </div>
        </footer>
      </div>
    </TooltipProvider>
  )
}

export default App
