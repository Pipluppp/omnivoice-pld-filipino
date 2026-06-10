import * as React from "react"

// Both the Vite dev server and Cloudflare Workers static assets (in
// single-page-application mode) answer missing paths with index.html,
// so a plain `res.ok` check is not enough — verify the content type too.
async function probe(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { method: "HEAD" })
    const type = res.headers.get("content-type") ?? ""
    return res.ok && !type.includes("text/html")
  } catch {
    return false
  }
}

/**
 * Probes the given URLs once and reports which ones resolve to real audio
 * files. URLs not yet probed are `undefined` (unknown).
 */
export function useAudioAvailability(urls: string[]) {
  const [available, setAvailable] = React.useState<Record<string, boolean>>({})
  const key = urls.join("\n")

  React.useEffect(() => {
    let cancelled = false
    for (const url of key.split("\n")) {
      if (!url) continue
      void probe(url).then((ok) => {
        if (!cancelled) {
          setAvailable((prev) => (prev[url] === ok ? prev : { ...prev, [url]: ok }))
        }
      })
    }
    return () => {
      cancelled = true
    }
  }, [key])

  return available
}
