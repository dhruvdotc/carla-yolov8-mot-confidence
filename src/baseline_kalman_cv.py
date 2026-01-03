# baseline_kalman_cv.py
# Constant-Velocity Kalman Filter baseline
# State: [pos, vel], Measurement: pos
# Fits on past, then rolls out future using the state estimate

import os
import numpy as np

DATASET_NPZ = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/dataset.npz"

DT = 1.0

# Tune-friendly defaults
R_POS = 1e-4
Q_VEL = 1e-6

P0_POS = 1e-3
P0_VEL = 1e-2


def ade_fde_np(pred, true):
    d = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    return float(np.mean(d)), float(d[-1])


def cv_matrices(dt, q_vel, r_pos):
    F = np.array([
        [1.0, dt],
        [0.0, 1.0],
    ], dtype=np.float64)

    H = np.array([[1.0, 0.0]], dtype=np.float64)

    Q = q_vel * np.array([
        [dt * dt, dt],
        [dt,      1.0],
    ], dtype=np.float64)

    R = np.array([[r_pos]], dtype=np.float64)
    return F, H, Q, R


def kf_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P


def kf_update_joseph(x, P, z, H, R):
    y = z - (H @ x)
    S = (H @ P @ H.T) + R
    K = (P @ H.T) @ np.linalg.inv(S)

    I = np.eye(P.shape[0])
    x = x + (K @ y)
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return x, P


def fit_cv_kf_1d(past_pos_1d, dt, q_vel, r_pos):
    F, H, Q, R = cv_matrices(dt, q_vel, r_pos)

    p0 = float(past_pos_1d[0])
    p1 = float(past_pos_1d[1]) if len(past_pos_1d) > 1 else p0
    v0 = (p1 - p0) / dt

    x = np.array([[p0], [v0]], dtype=np.float64)
    P = np.diag([P0_POS, P0_VEL]).astype(np.float64)

    for z in past_pos_1d:
        x, P = kf_predict(x, P, F, Q)
        x, P = kf_update_joseph(x, P, np.array([[float(z)]], dtype=np.float64), H, R)

    return x


def rollout_cv_mean_1d(x_end, future_len, dt):
    F, _, _, _ = cv_matrices(dt, Q_VEL, R_POS)
    x = x_end.copy()
    preds = []
    for _ in range(future_len):
        x = F @ x
        preds.append(float(x[0, 0]))
    return np.array(preds, dtype=np.float64)


def kalman_cv_predict_2d(past_xy, future_len):
    past_x = past_xy[:, 0]
    past_y = past_xy[:, 1]

    x_state = fit_cv_kf_1d(past_x, DT, Q_VEL, R_POS)
    y_state = fit_cv_kf_1d(past_y, DT, Q_VEL, R_POS)

    pred_x = rollout_cv_mean_1d(x_state, future_len, DT)
    pred_y = rollout_cv_mean_1d(y_state, future_len, DT)

    return np.stack([pred_x, pred_y], axis=1)


data = np.load(DATASET_NPZ, allow_pickle=True)
X = data["X"]
Y = data["Y"]
past_len = int(data["past"])
future_len = int(data["future"])

ades = []
fdes = []

for i in range(len(X)):
    pred = kalman_cv_predict_2d(X[i], future_len)
    ade, fde = ade_fde_np(pred, Y[i])
    ades.append(ade)
    fdes.append(fde)

print("DONE")
print("Windows evaluated:", len(X))
print("Past length:", past_len, "Future length:", future_len)
print("Kalman CV ADE:", float(np.mean(ades)))
print("Kalman CV FDE:", float(np.mean(fdes)))
print("Settings: R_POS =", R_POS, "Q_VEL =", Q_VEL)
