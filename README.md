# CricketVisionAI

AI cricket video analysis — ball detection, player roles (batsman / fielder / umpire / bowler), shot classification with pose, equipment detection, and a live dashboard.

**Author:** [Ali Raza Chaudhry](https://github.com/alirazach476)

<p align="center">
  <a href="#demo-video"><strong>Demo video</strong></a>
  &nbsp;·&nbsp;
  <a href="#results-gallery"><strong>Results</strong></a>
  &nbsp;·&nbsp;
  <a href="#ball-detection">Ball</a>
  &nbsp;·&nbsp;
  <a href="#shot-classification--pose">Shots</a>
  &nbsp;·&nbsp;
  <a href="#equipment--pitch">Equipment</a>
  &nbsp;·&nbsp;
  <a href="#player-roles">Players</a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Setup</a>
</p>

---

<a id="demo-video"></a>

## Demo video

https://github.com/alirazach476/Cricket-AI/raw/main/media/cricket.mp4

<video src="media/cricket.mp4" controls width="100%"></video>

[▶ Download / open `media/cricket.mp4`](./media/cricket.mp4)

---

<a id="results-gallery"></a>

## Results gallery

<a id="ball-detection"></a>

### Ball detection

<p align="center">
  <img src="media/ball-detection-predictions.png" width="100%" alt="Cricket ball detection predictions" />
</p>

<p align="center">
  <img src="media/ball-confusion-matrix.png" width="55%" alt="Ball detection normalized confusion matrix" />
</p>

<a id="shot-classification--pose"></a>

### Shot classification & pose

<p align="center">
  <img src="media/shot-pose-predictions-1.png" width="100%" alt="Cricket shot classification with pose keypoints" />
</p>

<p align="center">
  <img src="media/shot-pose-predictions-2.png" width="100%" alt="Additional shot + pose predictions" />
</p>

<p align="center">
  <img src="media/shot-confusion-matrix.png" width="70%" alt="Shot classification confusion matrix" />
</p>

<p align="center">
  <img src="media/shot-pose-training-results.png" width="100%" alt="Shot/pose training curves" />
</p>

<a id="equipment--pitch"></a>

### Equipment & pitch

<p align="center">
  <img src="media/equipment-detection-predictions.png" width="100%" alt="Bat, pad, stumps detection" />
</p>

<p align="center">
  <img src="media/equipment-detection-predictions-2.png" width="100%" alt="Equipment detection — additional batch" />
</p>

<p align="center">
  <img src="media/pitch-equipment-predictions.png" width="100%" alt="Pitch and equipment detection mosaic" />
</p>

<p align="center">
  <img src="media/equipment-confusion-matrix.png" width="60%" alt="Equipment detection confusion matrix" />
</p>

<p align="center">
  <img src="media/equipment-training-results.png" width="100%" alt="Equipment model training curves" />
</p>

<a id="player-roles"></a>

### Player roles

<p align="center">
  <img src="media/live-match-detection.png" width="100%" alt="Live match role detection — batsman, bowler, fielder, umpire" />
</p>

<p align="center">
  <img src="media/player-roles-confusion-matrix.png" width="60%" alt="Player roles confusion matrix" />
</p>

---

## Features

- Ball detection (`cricket_ball_detection/best.pt`)
- Batsman / fielder / umpire (`cricket detection/best.pt`)
- Bowler only (`Cricket Player Detection/best.pt`)
- Shot classification with pose (`Cricket Shot Classification`)
- Pitch / bat / pad (DRS weights)
- Live Analysis + Field Map HUD on annotated video
- FastAPI backend + React dashboard

---

<a id="quick-start"></a>

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
