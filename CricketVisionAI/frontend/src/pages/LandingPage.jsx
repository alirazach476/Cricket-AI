import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import GeometricBackdrop, { MotifRibbon } from '../components/GeometricBackdrop'
import SiteLogo from '../components/SiteLogo'
import UploadZone from '../components/UploadZone'
import ProcessingOverlay from '../components/ProcessingOverlay'
import { useEffect, useRef, useState } from 'react'
import { fetchJobStatus, processVideo, uploadVideo } from '../services/api'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

export default function LandingPage() {
  const navigate = useNavigate()
  const [uploadProgress, setUploadProgress] = useState(0)
  const [busy, setBusy] = useState(false)
  const [overlayLabel, setOverlayLabel] = useState('Uploading…')
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('upload')
  const [detail, setDetail] = useState('')
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const pollRef = useRef(null)
  const tickRef = useRef(null)
  const lastProgRef = useRef(0)

  const stopPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    if (!busy) {
      if (tickRef.current) clearInterval(tickRef.current)
      tickRef.current = null
      setElapsed(0)
      return undefined
    }
    const started = Date.now()
    tickRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000))
    }, 1000)
    return () => {
      if (tickRef.current) clearInterval(tickRef.current)
    }
  }, [busy])

  const applyStatus = (st) => {
    if (st.progress != null) {
      const p = Number(st.progress)
      lastProgRef.current = p
      setProgress(p)
    }
    if (st.stage) setStage(st.stage)
    if (st.message) {
      setDetail(st.message)
      if (st.stage === 'prepare') setOverlayLabel(st.message || 'Loading AI models…')
      else if (st.stage === 'analyze') setOverlayLabel('AI analyzing frames…')
      else if (st.stage === 'finalize') setOverlayLabel('Finishing reports…')
      else if (st.stage === 'done') setOverlayLabel('Done')
      else setOverlayLabel(st.message)
    }
  }

  const waitUntilDone = async (jobId) => {
    let misses = 0
    let relaunched = false
    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        const st = await fetchJobStatus(jobId)
        misses = 0
        applyStatus(st)
        if (st.job_status === 'completed') {
          return {
            job_id: st.job_id,
            input_video: st.input_video,
            output_video: st.output_video,
            artifacts: st.artifacts || {},
            stats: st.stats || {},
            status: 'success',
            message: st.message || 'Processing completed',
          }
        }
        if (st.job_status === 'failed') {
          throw new Error(st.message || 'Processing failed')
        }
        // Recovered after restart but not started again — relaunch once
        if (st.job_status === 'uploaded' && !relaunched) {
          relaunched = true
          setOverlayLabel('Restarting analysis…')
          setDetail('Server recovered your upload — starting again')
          await processVideo({ job_id: jobId, generate_highlights: true })
        }
      } catch (err) {
        const msg = err?.response?.data?.detail || err?.message || ''
        if (String(msg).startsWith('Processing failed')) {
          throw err instanceof Error ? err : new Error(String(msg))
        }
        if (
          err?.response?.status === 404 ||
          /Unknown job_id|No jobs yet/i.test(String(msg))
        ) {
          misses += 1
          if (misses >= 4) {
            throw new Error(
              'Job was lost after a server restart. Please upload the video again.',
            )
          }
          setDetail(`Reconnecting to server… (${misses}/4)`)
          // Try relaunching process — job store can recover the upload file
          if (!relaunched) {
            try {
              await processVideo({ job_id: jobId, generate_highlights: true })
              relaunched = true
              setOverlayLabel('Restarting analysis…')
            } catch {
              /* keep polling */
            }
          }
        } else {
          misses += 1
          if (misses >= 20) {
            throw new Error('Server not responding. Restart backend and upload again.')
          }
          setDetail('Waiting for server…')
        }
      }
      await sleep(900)
    }
  }

  const onFile = async (file) => {
    setError('')
    setBusy(true)
    setStage('upload')
    setOverlayLabel('Uploading video…')
    setProgress(2)
    setDetail(file.name)
    stopPoll()
    try {
      setUploadProgress(1)
      const up = await uploadVideo(file, (p) => {
        setUploadProgress(p)
        setProgress(Math.max(2, Math.min(12, p * 0.12)))
      })
      setUploadProgress(100)
      setProgress(12)
      setStage('prepare')
      setOverlayLabel('Starting analysis…')
      setDetail('Models load once, then frame-by-frame detection begins')

      await processVideo({
        job_id: up.job_id,
        generate_highlights: true,
      })

      const result = await waitUntilDone(up.job_id)
      setProgress(100)
      setOverlayLabel('Done — opening dashboard')
      setBusy(false)
      navigate('/dashboard', { state: { job: result } })
    } catch (err) {
      console.error(err)
      stopPoll()
      setBusy(false)
      setUploadProgress(0)
      setProgress(0)
      setError(err?.response?.data?.detail || err.message || 'Upload/process failed')
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      <GeometricBackdrop />
      <ProcessingOverlay
        visible={busy}
        label={overlayLabel}
        progress={progress}
        stage={stage}
        detail={
          detail
            ? `${detail}${elapsed ? ` · ${elapsed}s` : ''}`
            : elapsed
              ? `Working… ${elapsed}s`
              : ''
        }
      />

      <header className="relative z-10 mx-auto flex max-w-6xl items-center justify-between px-6 pb-4 pt-10">
        <div className="flex items-center gap-3">
          <SiteLogo size={56} showWordmark={false} />
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.35em] text-[#FF2D8D]">
              Smart Match Analytics
            </div>
            <h1 className="heading-glow text-2xl font-black uppercase tracking-wide md:text-3xl">
              Cricket Vision AI
            </h1>
          </div>
        </div>
        <div className="hidden rounded-full border border-white/15 bg-white/10 px-4 py-1.5 text-[11px] font-semibold uppercase tracking-wider backdrop-blur md:block">
          YOLO · ByteTrack · OCR
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-6xl px-6 pb-36 pt-4">
        <motion.section
          className="mx-auto max-w-3xl text-center"
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65 }}
        >
          <SiteLogo size={168} showWordmark={false} />
          <p className="mt-6 text-xs font-bold uppercase tracking-[0.4em] text-[#18B8FF]">
            Broadcast-Grade AI
          </p>
          <h2 className="heading-glow mt-2 text-4xl font-black uppercase leading-[1.05] md:text-6xl">
            Cricket Vision AI
          </h2>
          <p className="mt-2 text-2xl font-extrabold uppercase tracking-wide text-[#FF2D8D] md:text-3xl">
            Match Analytics System
          </p>
          <p className="mx-auto mt-4 max-w-xl text-base text-white/75 md:text-lg">
            Upload a clip — AI detects players & DRS objects, tracks the ball, classifies
            shots, and builds broadcast-style insights.
          </p>
          <div className="mx-auto mt-6 max-w-md overflow-hidden rounded-full">
            <MotifRibbon />
          </div>
        </motion.section>

        <div className="mt-10">
          <UploadZone
            onFile={onFile}
            uploadProgress={uploadProgress}
            disabled={busy}
          />
        </div>

        {error && (
          <p className="mx-auto mt-4 max-w-2xl rounded-xl bg-red-500/20 px-4 py-3 text-center text-sm text-red-200">
            {error}
          </p>
        )}

        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {[
            {
              t: 'Player + DRS Detection',
              d: 'Ball, Batsman, Bowler, Scoreboard, Bat, Pad, Pitch',
              c: '#FF2D8D',
            },
            {
              t: 'Tracking & Trajectory',
              d: 'ByteTrack IDs, live ball path panel, speed estimation',
              c: '#18B8FF',
            },
            {
              t: 'Shots & Scoreboard OCR',
              d: 'Cricket Actions YOLO: Cover Drive, Front Foot, Square Short, Helicopter, Pull',
              c: '#FFD233',
            },
          ].map((card) => (
            <div
              key={card.t}
              className="glass rounded-2xl p-5 transition hover:-translate-y-1 hover:shadow-[0_0_28px_rgba(255,45,141,0.25)]"
              style={{ borderColor: `${card.c}55` }}
            >
              <div className="chevron-band mb-4 rounded-sm opacity-90" />
              <h3 className="font-black uppercase tracking-wide" style={{ color: card.c }}>
                {card.t}
              </h3>
              <p className="mt-2 text-sm text-white/65">{card.d}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}
