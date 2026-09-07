import { motion } from 'framer-motion'

/** Original icon tiles — stars, drops, diamonds, flares (not ICC artwork) */
const TILES = [
  { bg: '#FF2D8D', icon: 'star', fg: '#2B0057' },
  { bg: '#18B8FF', icon: 'drop', fg: '#2B0057' },
  { bg: '#FFD233', icon: 'diamond', fg: '#2B0057' },
  { bg: '#4B0F8A', icon: 'flare', fg: '#FFD233' },
  { bg: '#FF2D8D', icon: 'petal', fg: '#FFFFFF' },
  { bg: '#18B8FF', icon: 'star', fg: '#FFFFFF' },
  { bg: '#FFD233', icon: 'drop', fg: '#2B0057' },
  { bg: '#3D1172', icon: 'diamond', fg: '#18B8FF' },
]

function MotifIcon({ type, color }) {
  if (type === 'star') {
    return (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill={color}>
        <path d="M12 2l2.4 7.2H22l-6 4.4 2.3 7.2L12 16.8 5.7 20.8 8 13.6 2 9.2h7.6z" />
      </svg>
    )
  }
  if (type === 'drop') {
    return (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill={color}>
        <path d="M12 2C12 2 5 11 5 15a7 7 0 0014 0C19 11 12 2 12 2z" />
      </svg>
    )
  }
  if (type === 'diamond') {
    return (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill={color}>
        <path d="M12 2l8 10-8 10L4 12z" />
      </svg>
    )
  }
  if (type === 'petal') {
    return (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill={color}>
        <circle cx="12" cy="8" r="4" />
        <circle cx="8" cy="14" r="4" />
        <circle cx="16" cy="14" r="4" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill={color}>
      <path d="M12 3l2 6 6 .5-4.5 4 1.5 6L12 16l-5 3.5 1.5-6L4 9.5 10 9z" />
    </svg>
  )
}

/** Horizontal mosaic ribbon like broadcast borders */
export function MotifRibbon({ className = '' }) {
  const row = [...TILES, ...TILES, ...TILES]
  return (
    <div className={`tile-ribbon ${className}`}>
      {row.map((t, i) => (
        <span
          key={`${t.icon}-${i}`}
          className="flex items-center justify-center"
          style={{ background: t.bg }}
        >
          <MotifIcon type={t.icon} color={t.fg} />
        </span>
      ))}
    </div>
  )
}

/** Layered V / chevron bands — pink, blue, yellow */
export function ChevronBands({ className = '' }) {
  return (
    <svg
      className={`w-full ${className}`}
      viewBox="0 0 1200 140"
      preserveAspectRatio="none"
      aria-hidden
    >
      <defs>
        <pattern id="dots" width="28" height="14" patternUnits="userSpaceOnUse">
          <circle cx="6" cy="7" r="3" fill="#2B0057" opacity="0.35" />
          <rect x="16" y="4" width="6" height="6" fill="#2B0057" opacity="0.25" />
        </pattern>
        <pattern id="gears" width="32" height="16" patternUnits="userSpaceOnUse">
          <circle cx="8" cy="8" r="5" fill="none" stroke="#0A3A66" strokeWidth="2" />
          <circle cx="24" cy="8" r="4" fill="#0A3A66" opacity="0.5" />
        </pattern>
        <pattern id="diamonds" width="24" height="14" patternUnits="userSpaceOnUse">
          <path d="M12 1l5 6-5 6-5-6z" fill="#C41E6A" opacity="0.7" />
        </pattern>
      </defs>
      <polygon points="0,140 0,40 600,5 1200,40 1200,140" fill="#FF2D8D" />
      <polygon points="0,140 0,40 600,5 1200,40 1200,140" fill="url(#dots)" />
      <polygon points="0,140 0,62 600,28 1200,62 1200,140" fill="#18B8FF" />
      <polygon points="0,140 0,62 600,28 1200,62 1200,140" fill="url(#gears)" />
      <polygon points="0,140 0,88 600,58 1200,88 1200,140" fill="#FFD233" />
      <polygon points="0,140 0,88 600,58 1200,88 1200,140" fill="url(#diamonds)" />
      <polygon points="0,140 0,112 600,92 1200,112 1200,140" fill="#FF2D8D" opacity="0.9" />
    </svg>
  )
}

/** Original brand mark — geometric stumps + ball (not a World Cup trophy copy) */
export function BrandMark({ size = 160 }) {
  return (
    <motion.div
      className="relative mx-auto"
      style={{ width: size, height: size }}
      animate={{ rotate: [0, 6, 0, -6, 0] }}
      transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
    >
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background:
            'conic-gradient(from 0deg, #FF2D8D 0 10deg, transparent 10deg 20deg, #FF2D8D 20deg 30deg, transparent 30deg 40deg)',
          mask: 'radial-gradient(farthest-side, transparent calc(100% - 14px), #000 calc(100% - 14px))',
          WebkitMask:
            'radial-gradient(farthest-side, transparent calc(100% - 14px), #000 calc(100% - 14px))',
        }}
      />
      <div className="absolute inset-[10%] flex items-center justify-center rounded-full bg-[#FF2D8D] shadow-[0_0_40px_rgba(255,45,141,0.45)]">
        <svg viewBox="0 0 80 80" className="h-[70%] w-[70%]" fill="none">
          <rect x="18" y="22" width="8" height="42" rx="2" fill="white" />
          <rect x="36" y="18" width="8" height="46" rx="2" fill="white" />
          <rect x="54" y="22" width="8" height="42" rx="2" fill="white" />
          <rect x="16" y="20" width="48" height="6" rx="2" fill="white" />
          <circle cx="64" cy="56" r="9" fill="#FFD233" stroke="white" strokeWidth="2" />
          <path
            d="M60 54c2 1 4 2 6 1M62 58c2 0 4 1 5 2"
            stroke="#2B0057"
            strokeWidth="1.2"
            strokeLinecap="round"
          />
        </svg>
      </div>
    </motion.div>
  )
}

export default function GeometricBackdrop() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute inset-0 motif-grid opacity-80" />

      <motion.div
        className="absolute -right-20 -top-20 h-[28rem] w-[28rem] rounded-full opacity-40"
        style={{
          background:
            'conic-gradient(from 0deg, #FF2D8D, transparent 25%, #18B8FF, transparent 55%, #FFD233, transparent 80%, #FF2D8D)',
          filter: 'blur(2px)',
        }}
        animate={{ rotate: 360 }}
        transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
      />

      <div className="absolute left-0 right-0 top-0 opacity-90">
        <MotifRibbon />
      </div>

      <div className="absolute bottom-0 left-0 right-0">
        <ChevronBands className="h-28 md:h-36 opacity-95" />
      </div>
    </div>
  )
}
