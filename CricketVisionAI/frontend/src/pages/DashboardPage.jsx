import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import ActionBar from '../components/ActionBar'
import AnnotatedPlayer from '../components/AnnotatedPlayer'
import {
  BallSpeedChart,
  DetectionCountsChart,
} from '../components/Charts'
import GeometricBackdrop, { MotifRibbon } from '../components/GeometricBackdrop'
import ProcessingOverlay from '../components/ProcessingOverlay'
import SiteLogo from '../components/SiteLogo'
import StatCard from '../components/StatCard'
import {
  annotatedVideoUrl,
  fetchAnalytics,
} from '../services/api'

export default function DashboardPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const [analytics, setAnalytics] = useState(null)
  const [videoSrc, setVideoSrc] = useState(annotatedVideoUrl())
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    setLoading(true)
    ;(async () => {
      try {
        const a = await fetchAnalytics()
        if (!mounted) return
        setAnalytics(a)
        setVideoSrc(annotatedVideoUrl())
        setError('')
      } catch (err) {
        if (!mounted) return
        setError(err?.response?.data?.detail || err.message || 'Failed to load analytics')
      } finally {
        if (mounted) setLoading(false)
      }
    })()
    return () => {
      mounted = false
    }
  }, [location.key])

  const live = analytics?.live || {}
  const job = location.state?.job

  return (
    <div className="relative min-h-screen overflow-hidden">
      <GeometricBackdrop />
      <ProcessingOverlay
        visible={loading}
        label="Loading match dashboard…"
        progress={null}
        stage="analyze"
        detail="Fetching analytics & annotated video"
      />

      <div className="relative z-10 mx-auto max-w-[1400px] px-4 py-6 md:px-6">
        <div className="mb-4 overflow-hidden rounded-xl">
          <MotifRibbon />
        </div>
        <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div className="flex items-center gap-4">
            <SiteLogo size={64} showWordmark={false} />
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.35em] text-[#FF2D8D]">
                Match Control
              </div>
              <h1 className="heading-glow text-3xl font-black uppercase tracking-wide">
                Cricket Vision AI Dashboard
              </h1>
              <p className="text-sm text-white/60">
                Job {analytics?.job_id || job?.job_id || '—'} ·{' '}
                {analytics?.job_status || 'ready'}
                {analytics?.stats?.expand_factor
                  ? ` · ${analytics.stats.expand_factor}× slow analysis`
                  : ''}
              </p>
            </div>
          </div>
          <ActionBar
            analytics={analytics}
            onUploadNew={() => navigate('/')}
          />
        </header>

        {error && (
          <div className="mb-4 rounded-xl bg-red-500/20 px-4 py-3 text-sm text-red-100">
            {error}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <section className="space-y-6">
            <AnnotatedPlayer
              src={videoSrc}
              fps={analytics?.stats?.playback_fps || analytics?.stats?.fps || 30}
            />

            <div className="grid gap-4 md:grid-cols-2">
              <BallSpeedChart series={analytics?.ball_speed?.series || []} />
              <DetectionCountsChart counts={analytics?.detection_counts || {}} />
            </div>

            <div className="glass rounded-2xl p-4">
              <h4 className="mb-2 font-semibold text-[#18B8FF]">Match Statistics</h4>
              <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
                <div>Frames: {analytics?.stats?.frames_processed ?? '—'}</div>
                <div>Tracking rows: {analytics?.stats?.tracking_rows ?? '—'}</div>
                <div>Highlights: {analytics?.stats?.highlight_count ?? '—'}</div>
              </div>
            </div>
          </section>

          <aside className="space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-white/60">
              Live Analytics
            </h3>
            <StatCard
              label="Ball Speed"
              value={
                live.ball_speed_kmh != null
                  ? `${Number(live.ball_speed_kmh).toFixed(0)} km/h`
                  : '—'
              }
              accent="#18B8FF"
              sub="Calibrated from pitch → km/h"
            />
            <StatCard label="Batsman" value={live.batsman} accent="#22C55E" />
            <StatCard label="Bowler" value={live.bowler} accent="#F97316" />
            <StatCard label="Fielder" value={live.fielder} accent="#FB923C" />
            <StatCard label="Umpire" value={live.umpire || 'Umpire'} accent="#EF4444" />
            <StatCard label="Runs" value={live.runs} accent="#FF2D8D" />
            <StatCard label="Wickets" value={live.wickets} accent="#FF2D8D" />
            <StatCard label="Overs" value={live.overs} accent="#A855F7" />
            <StatCard label="Current Run Rate" value={live.crr} accent="#18B8FF" />
            <StatCard label="Required Run Rate" value={live.rrr} accent="#FFD233" />
          </aside>
        </div>
      </div>
    </div>
  )
}
