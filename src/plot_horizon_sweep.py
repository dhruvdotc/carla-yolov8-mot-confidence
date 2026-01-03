# plot_horizon_sweep.py
# PURPOSE:
# Plot ADE and FDE vs prediction horizon (future frames) for:
# - CV baseline
# - Kalman CV
# - GRU-Abs
# - GRU-Delta
# Also annotates each point with number of windows (N) used at that horizon.
# Saves PNGs to outputs/bs_conf50_mot/plots/

import os
import matplotlib.pyplot as plt

# =========================
# OUTPUT PATH
# =========================

OUT_DIR = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# HORIZONS + WINDOW COUNTS
# =========================
# Past fixed at 20.
# N changes with horizon because longer future reduces valid windows.

future = [10, 20, 40, 60]
n_windows = [1999, 1687, 1303, 1098]

# =========================
# RESULTS (PAST=20)
# =========================

cv_ade = [0.0059733103147505354, 0.009139183282749125, 0.012761597201070113, 0.013785933681062451]
cv_fde = [0.012293708926904642, 0.02060684521398746, 0.029510846938612025, 0.03114084730787293]

kalman_cv_ade = [0.007226048986749614, 0.009673404219352939, 0.010340619140578032, 0.010321516592842715]
kalman_cv_fde = [0.01334565308324077, 0.02110774429620339, 0.024814691528331195, 0.024424491888654642]

gru_abs_ade = [0.01808746744479452, 0.022031364507797198, 0.020867533601668417, 0.03474172195756292]
gru_abs_fde = [0.027406651186339087, 0.03571063544102541, 0.027709771099747444, 0.056161142167556716]

gru_delta_ade = [0.01270416040480929, 0.014203318937380952, 0.011744887663089499, 0.012260874755770326]
gru_delta_fde = [0.02451931967390732, 0.029677147940387876, 0.02698604530673854, 0.022341562614563]

# =========================
# HELPERS
# =========================

def annotate_counts(xs, ys, counts):
    for x, y, n in zip(xs, ys, counts):
        plt.annotate(
            f"N={n}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=9,
        )

def save_show(fig_name):
    out_path = os.path.join(OUT_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)
    plt.show()

# =========================
# PLOT 1: ADE vs Horizon
# =========================

plt.figure()
plt.plot(future, cv_ade, marker="o", label="CV (deterministic)")
plt.plot(future, kalman_cv_ade, marker="o", label="Kalman CV")
plt.plot(future, gru_delta_ade, marker="o", label="GRU-Delta")
plt.plot(future, gru_abs_ade, marker="o", label="GRU-Abs")

annotate_counts(future, cv_ade, n_windows)

plt.xlabel("Future frames")
plt.ylabel("ADE")
plt.title("ADE vs Prediction Horizon (Past = 20)")
plt.grid(True)
plt.legend()

save_show("ade_vs_horizon.png")

# =========================
# PLOT 2: FDE vs Horizon
# =========================

plt.figure()
plt.plot(future, cv_fde, marker="o", label="CV (deterministic)")
plt.plot(future, kalman_cv_fde, marker="o", label="Kalman CV")
plt.plot(future, gru_delta_fde, marker="o", label="GRU-Delta")
plt.plot(future, gru_abs_fde, marker="o", label="GRU-Abs")

annotate_counts(future, cv_fde, n_windows)

plt.xlabel("Future frames")
plt.ylabel("FDE")
plt.title("FDE vs Prediction Horizon (Past = 20)")
plt.grid(True)
plt.legend()

save_show("fde_vs_horizon.png")

print("DONE")
