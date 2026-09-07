import { useEffect, useRef, useState } from 'react'

/** Plays annotated H.264 video (detections are burned into the file). */
export default function AnnotatedPlayer({ src, fps = 30 }) {
  const videoRef = useRef(null)
  const [time, setTime] = useState(0)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)

  useEffect(() => {
    setError('')
    setReady(false)
  }, [src])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const onTime = () => setTime(video.currentTime)
    const onReady = () => {
      setReady(true)
      setError('')
    }
    const onErr = () => {
      setError(
        'Could not play annotated video. Re-process the clip so it encodes as H.264 for the browser.',
      )
      setReady(false)
    }

    video.addEventListener('timeupdate', onTime)
    video.addEventListener('play', onTime)
    video.addEventListener('loadeddata', onReady)
    video.addEventListener('canplay', onReady)
    video.addEventListener('error', onErr)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('play', onTime)
      video.removeEventListener('loadeddata', onReady)
      video.removeEventListener('canplay', onReady)
      video.removeEventListener('error', onErr)
    }
  }, [src])

  return (
    <div className="glass relative overflow-hidden rounded-3xl">
      <div className="flex items-center justify-between gap-3 px-4 py-3">
        <div>
          <div className="text-[11px] font-bold uppercase tracking-[0.3em] text-[#FF2D8D]">
            Match playback
          </div>
          <h3 className="text-lg font-black uppercase tracking-wide">Annotated Video</h3>
        </div>
        <span className="rounded-full bg-white/10 px-3 py-1 text-[11px] text-white/70">
          {ready ? 'Ready' : error ? 'Unavailable' : 'Loading…'}
        </span>
      </div>
      <div className="chevron-band" />
      <div className="relative bg-black">
        <video
          ref={videoRef}
          key={src}
          src={src}
          controls
          playsInline
          preload="metadata"
          className="aspect-video w-full bg-black"
        >
          <source src={src} type="video/mp4" />
        </video>
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-6 text-center">
            <p className="max-w-md text-sm text-red-200">{error}</p>
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 text-xs text-white/50">
        <span>
          Playback {time.toFixed(1)}s · fps≈{fps}
        </span>
        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className="text-[#18B8FF] hover:underline"
        >
          Open in new tab
        </a>
      </div>
    </div>
  )
}
