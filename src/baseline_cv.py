# baseline_cv.py
# PURPOSE:
# Load dataset.npz (past → future trajectory windows)
# Predict the future using a simple Constant Velocity (CV) baseline
# Report ADE and FDE on the whole dataset

import os
import numpy as np

# =========================
# USER PATHS (EDIT IF NEEDED)
# =========================

DATASET_NPZ = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/dataset.npz"

# =========================
# FUNCTION DEFINITIONS
# =========================

def constant_velocity_predict(past_xy, future_len):
    # PURPOSE:
    # Predict future points using constant velocity
    #
    # INPUT:
    # past_xy   = shape (PAST_FRAMES, 2) containing [x,y] history
    # future_len = how many future frames to predict
    #
    # METHOD:
    # velocity = last_point - second_last_point
    # future[t] = last_point + velocity * (t+1)

    last = past_xy[-1]
    prev = past_xy[-2]
    vel = last - prev

    preds = []
    for i in range(future_len):
        step = i + 1
        preds.append(last + vel * step)

    return np.array(preds)


def compute_ade_fde(pred_xy, true_xy):
    # PURPOSE:
    # Compute standard trajectory prediction errors
    #
    # ADE = average distance error over all future timesteps
    # FDE = distance error at final future timestep

    diff = pred_xy - true_xy
    dists = np.sqrt((diff[:, 0] * diff[:, 0]) + (diff[:, 1] * diff[:, 1]))

    ade = float(np.mean(dists))
    fde = float(dists[-1])

    return ade, fde


# =========================
# MAIN SCRIPT LOGIC
# =========================

data = np.load(DATASET_NPZ, allow_pickle=True)

X = data["X"]   # shape: (num_windows, past, 2)
Y = data["Y"]   # shape: (num_windows, future, 2)

past_len = int(data["past"])
future_len = int(data["future"])

total = len(X)
if total == 0:
    raise RuntimeError("No windows found in dataset.npz. Your window builder produced 0 samples.")

ades = []
fdes = []

for i in range(total):
    past_xy = X[i]
    true_future = Y[i]

    pred_future = constant_velocity_predict(past_xy, future_len)

    ade, fde = compute_ade_fde(pred_future, true_future)
    ades.append(ade)
    fdes.append(fde)

ades = np.array(ades)
fdes = np.array(fdes)

print("DONE")
print("Windows evaluated:", total)
print("Past length:", past_len, "Future length:", future_len)
print("CV Baseline ADE:", float(np.mean(ades)))
print("CV Baseline FDE:", float(np.mean(fdes)))
