import * as React from "react"
import { Pause, Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { AudioScrubber } from "@/components/ui/waveform"
import { useWaveformPeaks } from "@/hooks/use-waveform-peaks"
import { cn } from "@/lib/utils"

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00"
  const s = Math.floor(seconds)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}

interface AudioPlayerProps {
  src: string
  /**
   * When the src changes (e.g. switching models mid-listen), carry the
   * playback position and play state over to the new clip so renditions
   * can be A/B compared at the same point in the utterance.
   */
  preservePosition?: boolean
  className?: string
}

// Both icons stay mounted and cross-fade (skill: contextual icon animations).
const ICON_CLASS =
  "absolute inset-0 m-auto size-4 transition-[opacity,scale,filter] duration-200 ease-[cubic-bezier(0.2,0,0,1)]"
const ICON_SHOWN = "scale-100 opacity-100 blur-none"
const ICON_HIDDEN = "scale-[0.25] opacity-0 blur-[4px]"

export function AudioPlayer({ src, preservePosition = false, className }: AudioPlayerProps) {
  const audioRef = React.useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = React.useState(false)
  const [duration, setDuration] = React.useState(0)
  const [time, setTime] = React.useState(0)
  const peaks = useWaveformPeaks(src)

  // Capture the outgoing clip's state during render, before the <audio>
  // element receives the new src.
  const carryRef = React.useRef<{ time: number; playing: boolean } | null>(null)
  const prevSrcRef = React.useRef(src)
  if (prevSrcRef.current !== src) {
    carryRef.current = preservePosition ? { time, playing } : null
    prevSrcRef.current = src
    if (!preservePosition) {
      // Reset display state for the fresh clip.
      queueMicrotask(() => {
        setTime(0)
        setPlaying(false)
      })
    }
  }

  // timeupdate only fires ~4×/s; animate the playhead between events.
  React.useEffect(() => {
    if (!playing) return
    let frame = requestAnimationFrame(function tick() {
      const audio = audioRef.current
      if (audio) setTime(audio.currentTime)
      frame = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(frame)
  }, [playing])

  const handleLoadedMetadata = () => {
    const audio = audioRef.current
    if (!audio) return
    setDuration(audio.duration)
    const carry = carryRef.current
    if (carry) {
      carryRef.current = null
      audio.currentTime = Math.min(carry.time, Math.max(audio.duration - 0.05, 0))
      // Autoplay can be blocked if the gesture is gone; the play button still works.
      if (carry.playing) audio.play().catch(() => {})
    }
  }

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      if (audio.ended) audio.currentTime = 0
      void audio.play()
    } else {
      audio.pause()
    }
  }

  const handleSeek = (value: number) => {
    const audio = audioRef.current
    if (!audio || !Number.isFinite(audio.duration)) return
    audio.currentTime = value
    setTime(value)
  }

  return (
    <div className={`flex items-center gap-3 ${className ?? ""}`}>
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <Button
        type="button"
        variant="outline"
        size="icon"
        onClick={togglePlay}
        aria-label={playing ? "Pause" : "Play"}
        className="relative size-10 rounded-full active:scale-[0.96]"
      >
        {/* translate-x nudges the triangle optically center */}
        <Play
          aria-hidden
          className={cn(ICON_CLASS, "translate-x-[1px]", playing ? ICON_HIDDEN : ICON_SHOWN)}
        />
        <Pause aria-hidden className={cn(ICON_CLASS, playing ? ICON_SHOWN : ICON_HIDDEN)} />
      </Button>
      <span className="w-10 text-right font-mono text-xs tabular-nums text-muted-foreground">
        {formatTime(time)}
      </span>
      <AudioScrubber
        data={peaks ?? []}
        currentTime={time}
        duration={duration}
        onSeek={handleSeek}
        className="h-12 flex-1"
      />
      <span className="w-10 font-mono text-xs tabular-nums text-muted-foreground">
        {formatTime(duration)}
      </span>
    </div>
  )
}
