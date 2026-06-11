import * as React from "react"

import { cn } from "@/lib/utils"

// Adapted from the ElevenLabs UI registry `waveform` component
// (https://ui.elevenlabs.io/docs/components/waveform), trimmed to the static
// waveform + scrubber this app needs. Mouse events were replaced with pointer
// capture so touch scrubbing works, and the random-data fallback with flat
// placeholder bars (shown while peaks decode).

interface WaveformProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Normalized 0..1 peaks. An empty array renders flat placeholder bars. */
  data?: number[]
  barWidth?: number
  barGap?: number
  barRadius?: number
  /** Bar height in px for silence; also the floor for quiet passages. */
  minBarHeight?: number
}

export function Waveform({
  data = [],
  barWidth = 3,
  barGap = 1,
  barRadius = 1,
  minBarHeight = 3,
  className,
  ...props
}: WaveformProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const containerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const render = () => {
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      const rect = container.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, rect.width, rect.height)

      const color =
        getComputedStyle(canvas).getPropertyValue("--foreground") || "#000"
      const barCount = Math.floor(rect.width / (barWidth + barGap))
      const centerY = rect.height / 2

      for (let i = 0; i < barCount; i++) {
        const value = data.length
          ? (data[Math.floor((i / barCount) * data.length)] ?? 0)
          : 0
        const barHeight = Math.max(minBarHeight, value * rect.height * 0.9)
        ctx.fillStyle = color
        ctx.globalAlpha = 0.25 + value * 0.75
        ctx.beginPath()
        ctx.roundRect(
          i * (barWidth + barGap),
          centerY - barHeight / 2,
          barWidth,
          barHeight,
          barRadius
        )
        ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    const observer = new ResizeObserver(render)
    observer.observe(container)
    // The bar color is sampled from --foreground at draw time, so repaint
    // when the theme class on <html> flips.
    const themeObserver = new MutationObserver(render)
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })
    render()
    return () => {
      observer.disconnect()
      themeObserver.disconnect()
    }
  }, [data, barWidth, barGap, barRadius, minBarHeight])

  return (
    <div ref={containerRef} className={cn("relative", className)} {...props}>
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  )
}

interface AudioScrubberProps extends WaveformProps {
  currentTime?: number
  duration?: number
  onSeek?: (time: number) => void
}

export function AudioScrubber({
  currentTime = 0,
  duration = 0,
  onSeek,
  className,
  ...waveformProps
}: AudioScrubberProps) {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = React.useState(false)
  const [localProgress, setLocalProgress] = React.useState(0)

  const progress = dragging
    ? localProgress
    : duration > 0
      ? Math.min(currentTime / duration, 1)
      : 0

  const seekTo = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect || duration <= 0) return
    const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1)
    setLocalProgress(ratio)
    onSeek?.(ratio * duration)
  }

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (duration <= 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setDragging(true)
    seekTo(event.clientX)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (duration <= 0) return
    const step = Math.max(0.1, duration / 20)
    if (event.key === "ArrowLeft") {
      event.preventDefault()
      onSeek?.(Math.max(0, currentTime - step))
    } else if (event.key === "ArrowRight") {
      event.preventDefault()
      onSeek?.(Math.min(duration, currentTime + step))
    }
  }

  return (
    <div
      ref={containerRef}
      role="slider"
      tabIndex={0}
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={duration}
      aria-valuenow={Math.min(currentTime, duration)}
      className={cn(
        "relative cursor-pointer touch-none select-none overflow-hidden rounded-md outline-none focus-visible:ring-3 focus-visible:ring-ring/30",
        className
      )}
      onPointerDown={handlePointerDown}
      onPointerMove={dragging ? (e) => seekTo(e.clientX) : undefined}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
      onKeyDown={handleKeyDown}
    >
      <Waveform className="h-full" {...waveformProps} />
      <div
        className="pointer-events-none absolute inset-y-0 left-0 bg-foreground/10"
        style={{ width: `${progress * 100}%` }}
      />
      <div
        className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-foreground"
        style={{ left: `${progress * 100}%` }}
      />
    </div>
  )
}
