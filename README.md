# CricketVisionAI

AI cricket video analysis — ball detection, player roles (batsman / fielder / umpire / bowler), speed estimation, and a live dashboard.

Built by [Ali Raza Chaudhry](https://github.com/alirazach476).

## Features

- Ball detection (`cricket_ball_detection/best.pt`)
- Batsman / fielder / umpire (`cricket detection/best.pt`)
- Bowler only (`Cricket Player Detection/best.pt`)
- Pitch / bat / pad (DRS weights)
- Live Analysis + Field Map HUD on annotated video
- FastAPI backend + React dashboard

## Quick start

### Backend

```bash
cd CricketVisionAI
pip install -r requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

### Frontend

```bash
cd CricketVisionAI/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173/

## Model layout

Weights are expected next to `CricketVisionAI/` (this repo root):

| Model | Path |
|-------|------|
| People | `cricket detection/best.pt` |
| Bowler | `Cricket Player Detection/best.pt` |
| Ball | `cricket_ball_detection/best.pt` |
| DRS | `CricketVisionAI/weights/drs_detection.pt` |

## License

Personal / portfolio project.
