# 1) take the tracking.csv table
# 2) convert it into ONE learning-ready dataset file (dataset.npz) containing past → future trajectory windows for ML models

import os
import numpy as np
import pandas as pd

# =========================
# USER PATHS (EDIT IF NEEDED)
# =========================

TRACKING_CSV = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/tracking.csv"
OUTPUT_NPZ   = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/dataset.npz"

# =========================
# WINDOW SETTINGS
# =========================

PAST_FRAMES   = 20   # number of past frames used as input (1s)
FUTURE_FRAMES = 10   # number of future frames used as prediction target (2s)

# =========================
# FUNCTION DEFINITIONS
# =========================

def build_windows_for_track(track_df):
    # PURPOSE:
    # Build sliding past to future windows for ONE track_id
    #
    # INPUT:
    # track_df = dataframe containing detections for a single track_id,
    #            already sorted by frame number
    #
    # OUTPUT:
    # X_list = list of past trajectories  (PAST_FRAMES x 2)
    # Y_list = list of future trajectories (FUTURE_FRAMES x 2)

    X_list = []
    Y_list = []

    frames = track_df["frame"].values
    xs = track_df["cx"].values
    ys = track_df["cy"].values

    total_len = len(track_df)
    window_size = PAST_FRAMES + FUTURE_FRAMES

    for start in range(0, total_len - window_size + 1):
        end = start + window_size

        window_frames = frames[start:end]

        # Check if frames are contiguous (no gaps)
        if not np.all(window_frames[1:] == window_frames[:-1] + 1):
            continue

        past_x = xs[start : start + PAST_FRAMES]
        past_y = ys[start : start + PAST_FRAMES]

        future_x = xs[start + PAST_FRAMES : end]
        future_y = ys[start + PAST_FRAMES : end]

        past = np.stack([past_x, past_y], axis=1)
        future = np.stack([future_x, future_y], axis=1)

        X_list.append(past)
        Y_list.append(future)

    return X_list, Y_list


# =========================
# MAIN SCRIPT LOGIC
# =========================

# Load tracking table
df = pd.read_csv(TRACKING_CSV)

# Sort by time so trajectories make sense
df = df.sort_values(["track_id", "frame"])

X_all = []
Y_all = []

# Process each track separately
for track_id in df["track_id"].unique():
    track_df = df[df["track_id"] == track_id]

    # Very short tracks cannot produce windows
    if len(track_df) < PAST_FRAMES + FUTURE_FRAMES:
        continue

    X_list, Y_list = build_windows_for_track(track_df)

    X_all.extend(X_list)
    Y_all.extend(Y_list)

# Convert lists to numpy arrays
X = np.array(X_all)
Y = np.array(Y_all)

# Save single dataset file
output_folder = os.path.dirname(OUTPUT_NPZ)
if output_folder != "" and not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.savez_compressed(
    OUTPUT_NPZ,
    X = X,
    Y = Y,
    past = PAST_FRAMES,
    future = FUTURE_FRAMES
)

print("DONE")
print("Total trajectory windows:", len(X))
print("X shape:", X.shape, " (past trajectories)")
print("Y shape:", Y.shape, " (future trajectories)")
print("Saved to:", OUTPUT_NPZ)
