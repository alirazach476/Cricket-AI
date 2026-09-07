import {
  ArrowDownTrayIcon,
  ArrowUpTrayIcon,
  DocumentArrowDownIcon,
  TableCellsIcon,
} from '@heroicons/react/24/outline'
import { jsPDF } from 'jspdf'
import { annotatedVideoUrl, csvDownloadUrls } from '../services/api'

export default function ActionBar({ onUploadNew, analytics }) {
  const downloadVideo = () => {
    const a = document.createElement('a')
    a.href = annotatedVideoUrl()
    a.download = 'annotated_output.mp4'
    a.click()
  }

  const downloadCsv = () => {
    const urls = csvDownloadUrls()
    Object.values(urls).forEach((url) => {
      const a = document.createElement('a')
      a.href = url
      a.target = '_blank'
      a.click()
    })
  }

  const downloadPdf = () => {
    const doc = new jsPDF()
    const live = analytics?.live || {}
    doc.setFillColor(43, 0, 87)
    doc.rect(0, 0, 210, 297, 'F')
    doc.setTextColor(255, 45, 141)
    doc.setFontSize(18)
    doc.text('Cricket Vision AI — Match Report', 14, 24)
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(12)
    const lines = [
      `Shot: ${live.shot_type ?? '—'}`,
      `Ball Speed: ${live.ball_speed_kmh ?? live.ball_speed_px_s ?? '—'}`,
      `Score: ${live.runs ?? '—'}/${live.wickets ?? '—'}`,
      `Overs: ${live.overs ?? '—'}`,
      `CRR: ${live.crr ?? '—'}  RRR: ${live.rrr ?? '—'}`,
      `Batsman: ${live.batsman ?? '—'}  Bowler: ${live.bowler ?? '—'}`,
    ]
    lines.forEach((t, i) => doc.text(t, 14, 44 + i * 10))
    doc.save('cricket_vision_ai_report.pdf')
  }

  const btn =
    'inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition hover:scale-[1.02]'

  return (
    <div className="flex flex-wrap gap-3">
      <button
        type="button"
        onClick={onUploadNew}
        className={`${btn} bg-gradient-to-r from-[#FF2D8D] to-[#4B0F8A]`}
      >
        <ArrowUpTrayIcon className="h-5 w-5" />
        Upload New Video
      </button>
      <button
        type="button"
        onClick={downloadVideo}
        className={`${btn} bg-[#18B8FF]/20 ring-1 ring-[#18B8FF]/50`}
      >
        <ArrowDownTrayIcon className="h-5 w-5" />
        Download Annotated Video
      </button>
      <button
        type="button"
        onClick={downloadCsv}
        className={`${btn} bg-[#FFD233]/15 ring-1 ring-[#FFD233]/40 text-[#FFD233]`}
      >
        <TableCellsIcon className="h-5 w-5" />
        Download CSV
      </button>
      <button
        type="button"
        onClick={downloadPdf}
        className={`${btn} bg-white/10 ring-1 ring-white/20`}
      >
        <DocumentArrowDownIcon className="h-5 w-5" />
        Download PDF
      </button>
    </div>
  )
}
