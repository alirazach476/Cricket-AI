import { CloudArrowUpIcon } from '@heroicons/react/24/solid'
import { motion } from 'framer-motion'
import { useCallback, useState } from 'react'

/** Drag & drop + click upload zone with progress. */
export default function UploadZone({ onFile, uploadProgress = 0, disabled }) {
  const [dragging, setDragging] = useState(false)

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0]
      if (!file) return
      if (!file.type.startsWith('video/') && !/\.(mp4|avi|mov|mkv|webm)$/i.test(file.name)) {
        alert('Please upload a cricket video file (mp4, avi, mov, mkv, webm).')
        return
      }
      onFile?.(file)
    },
    [onFile],
  )

  return (
    <motion.div
      className={`glass relative mx-auto w-full max-w-2xl rounded-3xl p-8 text-center transition ${
        dragging ? 'ring-2 ring-[#FF2D8D]' : ''
      } ${disabled ? 'opacity-60' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (!disabled) handleFiles(e.dataTransfer.files)
      }}
      whileHover={{ scale: disabled ? 1 : 1.01 }}
    >
      <div className="mb-5 overflow-hidden rounded-lg">
        <div className="chevron-band" />
      </div>
      <CloudArrowUpIcon className="mx-auto h-16 w-16 text-[#FF2D8D] drop-shadow-[0_0_12px_rgba(255,45,141,0.6)]" />
      <h3 className="mt-4 text-2xl font-black uppercase tracking-wide">Upload Cricket Video</h3>
      <p className="mt-2 text-sm text-white/70">
        Drag & drop a match clip, or click to browse
      </p>
      <label className="mt-6 inline-flex cursor-pointer items-center justify-center rounded-2xl bg-gradient-to-r from-[#FF2D8D] via-[#4B0F8A] to-[#18B8FF] px-8 py-3 text-lg font-black uppercase tracking-wide shadow-lg shadow-[#FF2D8D]/40 ring-1 ring-white/20">
        Choose Video
        <input
          type="file"
          accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
          className="hidden"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </label>

      {uploadProgress > 0 && uploadProgress < 100 && (
        <div className="mt-6">
          <div className="mb-1 flex justify-between text-xs text-white/70">
            <span>Uploading…</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-white/10">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-[#18B8FF] via-[#FF2D8D] to-[#FFD233]"
              initial={{ width: 0 }}
              animate={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}
    </motion.div>
  )
}
