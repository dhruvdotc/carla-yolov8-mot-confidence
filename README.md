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

## Quantitative Results (Part 1)

All metrics are computed without ground-truth annotations:

Part 1 uses self-consistency / temporal stability metrics from MOT outputs.

Part 2 evaluates forecasting error against the held-out future segment inside each trajectory window.

Detailed Part 1 figures and explanations:
**[`docs/figures/part1/README.md`](docs/figures/part1/README.md)**

### Summary Insights
- Increasing detection confidence reduces false positives but **increases short-lived track churn**.
- Aggressive confidence filtering lowers **scene occupancy** (fewer active objects per frame).
- Moderate confidence (0.50) provides the best trade-off between noise suppression and ID persistence.
- Downstream tracking behavior is driven primarily by **detection sparsity**, not tracker logic.


## Quantitative Results (Part 2) - Motion Forecasting Results (Past = 20)

This section evaluates short-horizon motion forecasting using trajectories extracted from the tracking outputs in Part 1.  
All experiments use a fixed past window of 20 frames, with prediction horizons of 10, 20, 40, and 60 frames.

Unlike Part 1, which analyzes detection–tracking stability, this section focuses on forecasting future motion using both physics-based and learning-based models. Since no ground-truth motion annotations are available, evaluation is performed using self-consistency error metrics (ADE and FDE) computed against observed future trajectories.

---

### Models Evaluated

- CV (deterministic): Constant-velocity extrapolation using the last two past positions.
- Kalman CV: Constant-velocity Kalman filter fit over the full past window, then rolled out.
- GRU-Abs: GRU predicts absolute future positions (x, y).
- GRU-Delta: GRU predicts relative displacements (dx, dy), which are integrated to obtain absolute positions.

---

### Metrics

- ADE (Average Displacement Error):  
  Mean Euclidean error averaged across all predicted future timesteps.

- FDE (Final Displacement Error):  
  Euclidean error at the final predicted timestep only.

(Lower values indicate better forecasting performance)

---

### Horizon Sweep - ADE (Past = 20)

| Future Frames | Windows (N) | CV ADE | Kalman CV ADE | GRU-Delta ADE | GRU-Abs ADE |
|--------------|-------------|--------|---------------|---------------|-------------|
| 10 | 1999 | 0.00597 | 0.00723 | 0.01270 | 0.01809 |
| 20 | 1687 | 0.00914 | 0.00967 | 0.01420 | 0.02203 |
| 40 | 1303 | 0.01276 | 0.01034 | 0.01174 | 0.02087 |
| 60 | 1098 | 0.01379 | 0.01032 | 0.01226 | 0.03474 |

---

### Horizon Sweep - FDE (Past = 20)

| Future Frames | Windows (N) | CV FDE | Kalman CV FDE | GRU-Delta FDE | GRU-Abs FDE |
|--------------|-------------|--------|---------------|---------------|-------------|
| 10 | 1999 | 0.01229 | 0.01335 | 0.02452 | 0.02741 |
| 20 | 1687 | 0.02061 | 0.02111 | 0.02968 | 0.03571 |
| 40 | 1303 | 0.02951 | 0.02481 | 0.02699 | 0.02771 |
| 60 | 1098 | 0.03114 | 0.02442 | 0.02234 | 0.05616 |

Detailed Part 2 figures and explanations:
**[`docs/figures/part2/README.md`](docs/figures/part2/README.md)**
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
│   │    ├── part1/
│   │    │   ├── README.md
│   │    │   ├── track_lifetime_histogram.png
│   │    │   ├── short_track_churn_lt1s.png
│   │    │   ├── avg_active_tracks_per_frame.png
│   │    │   └── avg_detections_per_frame.png
│   │    ├── part2/
│   │    │   ├── README.md
│   │    │   ├── ade_vs_horizon.png        # (add)
│   │    │   └── fde_vs_horizon.png        # (add)
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
