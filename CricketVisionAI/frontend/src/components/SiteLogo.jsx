/** Original Cricket Vision AI mark — T20-inspired energy style, not an ICC logo */
export default function SiteLogo({ size = 120, showWordmark = true }) {
  const s = size
  return (
    <div className="inline-flex flex-col items-center gap-2">
      <svg
        width={s}
        height={s}
        viewBox="0 0 200 200"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="Cricket Vision AI"
        role="img"
      >
        {/* Jagged energy ring (original geometry) */}
        <path
          fill="#FF2D8D"
          d="M100 8l12 22 24-8-2 26 26 6-16 20 18 18-24 6 2 26-24-10-12 24-12-24-24 10 2-26-24-6 18-18-16-20 26-6-2-26 24 8z"
        />
        <circle cx="100" cy="100" r="58" fill="#0A1628" />
        {/* Stylized CVA / cricket sweep — original */}
        <text
          x="100"
          y="112"
          textAnchor="middle"
          fill="#FF2D8D"
          fontFamily="Segoe UI, Arial Black, sans-serif"
          fontWeight="900"
          fontSize="42"
          letterSpacing="-2"
        >
          CV
        </text>
        <circle cx="148" cy="128" r="10" fill="#18B8FF" />
        <path
          d="M142 126c3 1 6 3 8 2"
          stroke="#0A1628"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
      {showWordmark && (
        <div className="text-center leading-tight">
          <div className="text-[10px] font-bold uppercase tracking-[0.35em] text-[#FF2D8D]">
            Cricket Vision AI
          </div>
          <div className="text-xs font-black uppercase tracking-wide text-white/90">
            Smart Match Analytics
          </div>
        </div>
      )}
    </div>
  )
}
