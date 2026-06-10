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

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-sm tabular-nums">{value}</span>
    </div>
  )
}

function ModelOutputs({
  sample,
  availability,
}: {
  sample: Sample
  availability: Record<string, boolean>
}) {
  const [modelId, setModelId] = React.useState(MODELS[0].id)

  const isAvailable = (model: ModelInfo) =>
    availability[modelUrl(model, sample)] === true
  const availableModels = MODELS.filter(isAvailable)
  const selected =
    availableModels.find((m) => m.id === modelId) ?? availableModels[0]

  // Number keys 1-4 jump between available model outputs.
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (event.target instanceof HTMLElement && event.target.closest("input, textarea")) return
      const index = Number.parseInt(event.key, 10) - 1
      const model = MODELS[index]
      if (model && availability[modelUrl(model, sample)]) {
        setModelId(model.id)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [availability, sample])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model outputs</CardTitle>
        <CardDescription>
          Voice clones of the reference speaker reading the same line. Switch
          models mid-playback to compare them at the same point in the
          utterance.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <ToggleGroup
          type="single"
          variant="outline"
          spacing={0}
          value={selected?.id ?? ""}
          onValueChange={(value) => value && setModelId(value)}
          className="w-full max-w-full overflow-x-auto"
        >
          {MODELS.map((model, index) => {
            const available = isAvailable(model)
            const item = (
              <ToggleGroupItem
                key={model.id}
                value={model.id}
                disabled={!available}
                className="flex-1"
              >
                <span>{model.label}</span>
                <Kbd className="ml-1.5 hidden sm:inline-flex">{index + 1}</Kbd>
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

        {selected ? (
          <>
            <AudioPlayer src={modelUrl(selected, sample)} preservePosition />
            <Separator />
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
              <span className="text-xs text-muted-foreground">{selected.detail}</span>
              <div className="flex flex-wrap gap-x-6 gap-y-2">
                <MetricChip
                  label="WER"
                  value={`${selected.metrics.wer.toFixed(2)}%`}
                />
                <MetricChip label="SIM-o" value={selected.metrics.simO.toFixed(3)} />
                <MetricChip label="UTMOS" value={selected.metrics.utmos.toFixed(2)} />
              </div>
            </div>
          </>
        ) : (
          <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
            Generated samples for this utterance haven&apos;t been added yet.
            Drop them in{" "}
            <code className="font-mono text-xs">
              public/audio/&lt;model&gt;/{sample.id}.wav
            </code>{" "}
            and they&apos;ll appear here automatically.
          </div>
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
        <CardDescription>
          Measured on the full PLD Filipino test split. Lower WER is better;
          higher SIM-o (speaker similarity) and UTMOS (naturalness) are better.
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
                  Listening tests: base checkpoint vs. three fine-tuned
                  learning-rate variants
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
                      onClick={() => setSampleId(s.id)}
                      aria-current={active ? "true" : undefined}
                      className={`w-full rounded-lg border px-3 py-2.5 text-left transition-colors ${
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
            <section>
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
              <blockquote className="text-pretty text-lg leading-relaxed">
                “{sample.text}”
              </blockquote>
            </section>

            <Card>
              <CardHeader>
                <CardTitle>Speaker audio</CardTitle>
                <CardDescription>
                  What the models were given, and what the real speaker
                  actually sounds like reading the line.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-5">
                <div className="flex flex-col gap-2">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <h3 className="text-sm font-medium">Voice prompt</h3>
                    <span className="text-xs text-muted-foreground">
                      the cloning input the models heard — a different line
                      from the same speaker
                    </span>
                  </div>
                  <AudioPlayer src={promptUrl(sample)} />
                  <p className="text-xs italic text-muted-foreground">
                    “{sample.promptText}”
                  </p>
                </div>
                <Separator />
                <div className="flex flex-col gap-2">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <h3 className="text-sm font-medium">Ground truth</h3>
                    <span className="text-xs text-muted-foreground">
                      the real recording of the line above
                    </span>
                  </div>
                  <AudioPlayer src={referenceUrl(sample)} />
                </div>
              </CardContent>
            </Card>

            <ModelOutputs sample={sample} availability={availability} />

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
