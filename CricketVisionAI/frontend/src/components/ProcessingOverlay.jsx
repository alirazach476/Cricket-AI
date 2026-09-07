import { motion } from 'framer-motion'

const STAGE_HINTS = {
  upload: 'Sending your clip to the server',
  prepare: 'Loading YOLO models… progress should move past this quickly',
  analyze: 'YOLO · DRS · ByteTrack · Shot AI · OCR',
  finalize: 'Writing reports, pitch map & highlights',
  done: 'Opening your dashboard',
}

/** Premium sports-broadcast style processing animation with live progress. */
export default function ProcessingOverlay({
  visible,
  label = 'AI Processing…',
  progress = null,
  stage = '',
  detail = '',
}) {
  if (!visible) return null
  const pct = progress != null ? Math.max(0, Math.min(100, Number(progress))) : null
  const hint = STAGE_HINTS[stage] || detail || 'Player YOLO · DRS · ByteTrack · Shot AI · Trajectory · OCR'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#2B0057]/85 backdrop-blur-md">
      <div className="glass-strong relative w-[min(92vw,440px)] rounded-3xl p-8 text-center shadow-2xl shadow-[#FF2D8D]/20">
        <motion.div
          className="mx-auto h-28 w-28 rounded-full border-4 border-[#FF2D8D]/40 border-t-[#FF2D8D] border-r-[#18B8FF]"
          animate={{ rotate: 360 }}
          transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute left-1/2 top-[4.6rem] h-10 w-10 -translate-x-1/2 rounded-full bg-[#FFD233]/90"
          animate={{ scale: [1, 1.15, 1], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 1.4, repeat: Infinity }}
        />
        <h3 className="mt-8 text-xl font-bold tracking-wide">{label}</h3>
        <p className="mt-2 text-sm text-white/70">{hint}</p>

        {pct != null && (
          <div className="mt-5">
            <div className="mb-1 flex justify-between text-xs font-semibold text-white/80">
              <span className="capitalize">{stage || 'working'}</span>
              <span>{pct.toFixed(0)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-white/10">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-[#18B8FF] via-[#FF2D8D] to-[#FFD233]"
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ type: 'spring', stiffness: 120, damping: 20 }}
              />
            </div>
            {detail ? (
              <p className="mt-2 truncate text-[11px] text-white/55">{detail}</p>
            ) : null}
          </div>
        )}

        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {['Upload', 'Load models', 'Detect', 'Analyze'].map((step, i) => (
            <motion.span
              key={step}
              className="rounded-full bg-white/10 px-3 py-1 text-xs"
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 1.6, repeat: Infinity, delay: i * 0.25 }}
            >
              {step}
            </motion.span>
          ))}
        </div>
      </div>
    </div>
  )
}
