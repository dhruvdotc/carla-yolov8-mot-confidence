# CARLA Multi-Object Tracking: Detection Confidence vs Temporal Stability

This project studies how **detection confidence thresholds propagate through a multi-object tracking pipeline**, affecting temporal stability, track persistence, and scene coverage.  
All experiments are conducted in the **CARLA simulator** using YOLO-based detection and online tracking.

The focus is not benchmark accuracy, but **system-level behavior** under realistic conditions (domain shift, occlusion, sparse detections).

![Online multi-object tracking with persistent IDs](docs/media/tracking.gif)
https://www.youtube.com/watch?v=ls9mTG2TMYw
---

## Project Pipeline
```text
CARLA Simulation
↓
RGB Camera Capture
↓
YOLO Detection (varying confidence thresholds)
↓
Online Tracking (default tracker / BoT-SORT)
↓
MOT-style Labels
↓
Temporal Stability Analysis
```

---

## Key Questions Explored

- How does detection confidence affect **track lifetime**?
- Does higher confidence reduce noise at the cost of **temporal continuity**?
- Can tracking mitigate detection sparsity?
- How do different tracking configurations behave under identical conditions?

---

## Quantitative Results

All metrics are computed **without ground-truth annotations**, using self-consistency measures derived from MOT-style outputs.

Detailed figures and explanations are available here:  
**[`docs/figures/FIGREADME.md`](docs/figures/FIGREADME.md)**

### Summary Insights
- Increasing detection confidence reduces false positives but **increases short-lived track churn**.
- Aggressive confidence filtering lowers **scene occupancy** (fewer active objects per frame).
- Moderate confidence (0.50) provides the best trade-off between noise suppression and ID persistence.
- Downstream tracking behavior is driven primarily by **detection sparsity**, not tracker logic.

---

## Repository Structure
```text
├── src/
│ ├── stage0_camera.py # CARLA data capture (ego vehicle + camera)
│ ├── export_mot_labels.py # Convert detections/tracks to MOT-style labels
│ └── evaluate_tracking_runs.py # Temporal stability metrics + plots
│
├── docs/
│ └── figures/
│ ├── FIGREADME.md
│ ├── track_lifetime_histogram.png
│ ├── short_track_churn_lt1s.png
│ ├── avg_active_tracks_per_frame.png
│ └── avg_detections_per_frame.png
│
└── README.md
```

---

## How to Reproduce (High Level)

1. **Capture data**
   ```bash
   python src/stage0_camera.py
   ```
2. **Run detection / tracking** : Generate detection-only and tracking outputs at different confidence thresholds.

3. **Export MOT-style labels**
   ```bash
   python src/export_mot_labels.py
   ```


4. **Evaluate temporal stability**
   ```bash
   python src/evaluate_tracking_runs.py
   ```
