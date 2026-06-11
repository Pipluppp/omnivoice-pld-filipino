import * as React from "react"

const PEAK_BUCKETS = 128

// Decoding is per-URL and the wavs never change within a session, so cache
// the promises: toggling back to an already-heard model redraws instantly.
const cache = new Map<string, Promise<number[] | null>>()

async function decodePeaks(url: string): Promise<number[] | null> {
  try {
    const res = await fetch(url)
    // Missing paths come back as index.html with HTTP 200 (SPA fallback),
    // same as in use-audio-availability — check the content type.
    const type = res.headers.get("content-type") ?? ""
    if (!res.ok || type.includes("text/html")) return null

    const encoded = await res.arrayBuffer()
    const audio = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(
      encoded
    )
    const channel = audio.getChannelData(0)
    const bucketSize = Math.max(1, Math.floor(channel.length / PEAK_BUCKETS))
    const stride = Math.max(1, Math.floor(bucketSize / 64))

    const peaks: number[] = []
    for (let i = 0; i < PEAK_BUCKETS; i++) {
      const start = i * bucketSize
      const end = Math.min(start + bucketSize, channel.length)
      let peak = 0
      for (let j = start; j < end; j += stride) {
        const v = Math.abs(channel[j])
        if (v > peak) peak = v
      }
      peaks.push(peak)
    }

    const max = Math.max(...peaks)
    return max > 0 ? peaks.map((p) => p / max) : peaks
  } catch {
    return null
  }
}

/**
 * Normalized 0..1 waveform peaks for an audio URL, or `null` while decoding
 * (and for files that don't exist yet).
 */
export function useWaveformPeaks(url: string) {
  // The result is tagged with its URL so a stale value is never shown for a
  // freshly-switched src (it reads as null until its own decode lands).
  const [result, setResult] = React.useState<{
    url: string
    peaks: number[] | null
  } | null>(null)

  React.useEffect(() => {
    let cancelled = false
    let promise = cache.get(url)
    if (!promise) {
      promise = decodePeaks(url)
      cache.set(url, promise)
    }
    void promise.then((peaks) => {
      if (!cancelled) setResult({ url, peaks })
    })
    return () => {
      cancelled = true
    }
  }, [url])

  return result?.url === url ? result.peaks : null
}
