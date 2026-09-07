export default function StatCard({ label, value, accent = '#FF2D8D', sub }) {
  return (
    <div
      className="glass relative overflow-hidden rounded-2xl p-4 transition hover:-translate-y-0.5 hover:shadow-[0_0_24px_rgba(255,45,141,0.2)]"
      style={{ boxShadow: `inset 3px 0 0 ${accent}` }}
    >
      <div
        className="pointer-events-none absolute -right-6 -top-6 h-16 w-16 rounded-full opacity-20"
        style={{ background: accent }}
      />
      <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/55">
        {label}
      </div>
      <div className="mt-1 text-2xl font-black tracking-tight" style={{ color: accent }}>
        {value ?? '—'}
      </div>
      {sub ? <div className="mt-1 text-xs text-white/50">{sub}</div> : null}
    </div>
  )
}
