# Cricket Vision AI – Frontend

React + Vite dashboard for Smart Match Analytics.

## Run

```bash
# Terminal 1 — API (from CricketVisionAI/)
pip install -r backend/requirements.txt
pip install easyocr ultralytics opencv-python tensorflow
uvicorn app.main:app --app-dir backend --reload --port 8000

# Terminal 2 — UI
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

## Theme

Original sports-broadcast palette inspired by CWC 2023 colors (no official logos/artwork):
`#2B0057` · `#4B0F8A` · `#FF2D8D` · `#18B8FF` · `#FFD233` · cards `#3D1172`
