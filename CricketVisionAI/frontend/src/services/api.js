import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  timeout: 1000 * 60 * 60, // long process
})

export async function uploadVideo(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (!onProgress || !evt.total) return
      onProgress(Math.round((evt.loaded / evt.total) * 100))
    },
  })
  return data
}

export async function processVideo(payload) {
  const { data } = await api.post('/process', payload)
  return data
}

export async function fetchJobStatus(jobId) {
  const path = jobId ? `/job/${jobId}` : '/job'
  const { data } = await api.get(path)
  return data
}

export async function fetchAnalytics() {
  const { data } = await api.get('/analytics')
  return data
}

export async function fetchTrajectory() {
  const { data } = await api.get('/trajectory')
  return data
}

export async function fetchReports() {
  const { data } = await api.get('/reports')
  return data
}

export async function fetchHighlights() {
  const { data } = await api.get('/highlights')
  return data
}

export function annotatedVideoUrl() {
  // Prefer static mount (reliable range requests) with cache-bust
  return `/static/output/annotated_output.mp4?t=${Date.now()}`
}

export function annotatedVideoApiUrl() {
  return `/annotated-video?t=${Date.now()}`
}

export function csvDownloadUrls() {
  return {
    tracking: '/static/output/tracking.csv',
    analytics: '/static/output/analytics.csv',
    shots: '/static/output/shots.csv',
  }
}

export default api
