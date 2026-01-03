# CARLA Multi-Object Tracking: Detection Confidence vs Temporal Stability (+ Motion Forecasting)

This repo studies how **detection confidence thresholds propagate through an online multi-object tracking (MOT) pipeline**, and how downstream tracking outputs can be reused to build a **short-horizon motion forecasting** dataset (past → future trajectories).

All experiments are from **CARLA** RGB camera footage using YOLO-based detection and online tracking.

![Online multi-object tracking with persistent IDs](docs/media/tracking.gif)  
https://www.youtube.com/watch?v=ls9mTG2TMYw

---

## Project Pipeline (Part 1 → Part 2)

```text
CARLA Simulation
↓
RGB Camera Capture
↓
YOLO Detection (varying confidence thresholds)
↓
Online Tracking (ByteTrack / BoT-SORT)
↓
MOT-style Label Export (per-frame .txt)
↓
PART 1: Temporal Stability Analysis (track lifetime, churn, scene coverage)
↓
PART 2: Motion Forecasting Dataset (tracking.csv → windowed trajectories)
↓
Baselines + GRU models + Horizon Sweep plots
```

---

## Key Questions Explored

Part 1: Tracking stability

- How does detection confidence affect track lifetime and ID persistence?
- Does higher confidence reduce noise at the cost of temporal continuity?
- Does tracking mitigate detection sparsity, or does sparsity dominate behavior?

Part 2: Motion forecasting (from tracking outputs)

- Given past = 20 frames (~1s) of tracked center points (cx, cy), predict the next T future frames:
- How strong is a simple physics baseline (CV)?
- Does a Kalman filter (CV model) improve long-horizon stability by denoising velocity estimates?
- Can a small GRU beat physics when trained on tracker-derived trajectories?
- How do errors scale as prediction horizon increases?
Note: (cx, cy) are YOLO-normalized image coordinates. Errors (ADE/FDE) are in normalized units.

---

## Quantitative Results

All metrics are computed without ground-truth annotations:

Part 1 uses self-consistency / temporal stability metrics from MOT outputs.

Part 2 evaluates forecasting error against the held-out future segment inside each trajectory window.

Detailed Part 1 figures and explanations:
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
│   ├── stage0_camera.py
│   ├── export_mot_labels.py
│   ├── evaluate_tracking_runs.py
│   ├── build_table_tracker.py        # labels/ -> tracking.csv
│   ├── make_windows.py               # tracking.csv -> dataset.npz (past/future windows)
│   ├── baseline_cv.py                # deterministic CV baseline
│   ├── baseline_kalman_cv.py         # Kalman CV baseline
│   ├── train_gru.py                  # GRU absolute future
│   ├── train_gru_deltas.py           # GRU delta future
│   └── plot_horizon_sweep.py         # ADE/FDE vs horizon plots
│
├── docs/
│   ├── figures/
│   │   ├── FIGREADME.md
│   │   ├── track_lifetime_histogram.png
│   │   ├── short_track_churn_lt1s.png
│   │   ├── avg_active_tracks_per_frame.png
│   │   ├── avg_detections_per_frame.png
│   │   ├── ade_vs_horizon.png        # (add)
│   │   └── fde_vs_horizon.png        # (add)
│   └── media/
│       └── tracking.gif
│
└── README.md
```

---

## How to Reproduce (High Level)

Part 1: Detection + tracking + stability metrics

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

Part 2: Motion forecasting from tracking outputs

1. **Merge label files into just one table**
   ```bash
   python src/build_table_tracker.py
   ```
2. **Build past→future windows** : May tweak / trial PAST_FRAMES and FUTURE_FRAMES in make_windows.py
   ```bash
   python src/make_windows.py
   ```

3. **Run baselines**
   ```bash
   python src/baseline_cv.py
   python src/baseline_kalman_cv.py
   ```
   
4. **Train GRU models**
   ```bash
   python src/train_gru.py
   python src/train_gru_deltas.py
   ```
5. **Plot horizon sweep**
   ```bash
   python src/plot_horizon_sweep.py
   ```

**Notes / Limitations**
- Tracking trajectories come from detector + tracker outputs, not human labeled ground truth.
- Therefore Part 2 evaluates forecasting consistency w.r.t. tracker-derived motion, not real world position accuracy.
