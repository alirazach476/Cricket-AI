import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from 'recharts'

const COLORS = ['#FF2D8D', '#18B8FF', '#FFD233', '#A855F7', '#34D399', '#F97316']

export function BallSpeedChart({ series = [] }) {
  const data = series
    .map((p) => ({
      t: Number(p.t?.toFixed?.(1) ?? p.t),
      speed: p.speed_kmh != null ? Number(p.speed_kmh) : null,
    }))
    .filter((p) => p.speed != null && !Number.isNaN(p.speed))
  return (
    <div className="glass h-64 rounded-2xl p-4">
      <h4 className="mb-2 text-sm font-semibold text-[#18B8FF]">Ball Speed (km/h)</h4>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey="t" stroke="#aaa" tick={{ fontSize: 10 }} />
          <YAxis
            stroke="#aaa"
            tick={{ fontSize: 10 }}
            domain={[0, 180]}
            label={{ value: 'km/h', angle: -90, position: 'insideLeft', fill: '#aaa' }}
          />
          <Tooltip
            contentStyle={{ background: '#3D1172', border: '1px solid #FF2D8D' }}
            formatter={(v) => [`${Number(v).toFixed(1)} km/h`, 'Speed']}
          />
          <Line type="monotone" dataKey="speed" stroke="#FF2D8D" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ShotDistributionChart({ distribution = {} }) {
  const data = Object.entries(distribution).map(([name, value]) => ({ name, value }))
  return (
    <div className="glass h-64 rounded-2xl p-4">
      <h4 className="mb-2 text-sm font-semibold text-[#FFD233]">Shot Distribution</h4>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey="name" stroke="#aaa" tick={{ fontSize: 9 }} interval={0} angle={-20} textAnchor="end" height={60} />
          <YAxis stroke="#aaa" tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ background: '#3D1172', border: '1px solid #FFD233' }} />
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function DetectionCountsChart({ counts = {} }) {
  const data = Object.entries(counts).map(([name, value]) => ({ name, value }))
  return (
    <div className="glass h-64 rounded-2xl p-4">
      <h4 className="mb-2 text-sm font-semibold text-[#FF2D8D]">Detection Counts</h4>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" />
          <XAxis type="number" stroke="#aaa" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" stroke="#aaa" tick={{ fontSize: 10 }} width={80} />
          <Tooltip contentStyle={{ background: '#3D1172', border: '1px solid #18B8FF' }} />
          <Bar dataKey="value" fill="#18B8FF" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function BallTimeline({ shots = [] }) {
  return (
    <div className="glass rounded-2xl p-4">
      <h4 className="mb-3 text-sm font-semibold text-white">Ball Timeline · Shots</h4>
      <div className="flex max-h-40 flex-col gap-2 overflow-y-auto">
        {shots.length === 0 && (
          <p className="text-sm text-white/50">No shot events yet.</p>
        )}
        {shots.slice(0, 40).map((s, i) => (
          <div
            key={`${s.frame_number}-${i}`}
            className="flex items-center justify-between rounded-xl bg-white/5 px-3 py-2 text-sm"
          >
            <span className="font-medium text-[#FFD233]">{s.shot_name}</span>
            <span className="text-white/60">
              t={Number(s.timestamp_sec).toFixed(1)}s · {(s.confidence * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
