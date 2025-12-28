import os
import re
import csv
from collections import defaultdict
from statistics import mean, median
import matplotlib.pyplot as plt

BASE_OUT = r"C:\Users\dhruv\carla_sim\outputs"
FPS = 10  # desired frame detectin fps (to achieve desired 4500 frames)

RUNS = {
    "bt_conf40_mot": r"C:\Users\dhruv\carla_sim\outputs\bt_conf40_mot\labels",
    "bt_conf50_mot": r"C:\Users\dhruv\carla_sim\outputs\bt_conf50_mot\labels",
    "bt_conf65_mot": r"C:\Users\dhruv\carla_sim\outputs\bt_conf65_mot\labels",
}

REPORT_DIR = os.path.join(BASE_OUT, "_report_tracking")
os.makedirs(REPORT_DIR, exist_ok=True)


def frame_index_from_name(fname: str) -> int:
    m = re.search(r"(\d+)", fname)
    return int(m.group(1)) if m else -1


def parse_labels_folder(labels_dir: str):
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    if not files:
        raise RuntimeError(f"No .txt label files in: {labels_dir}")

    det_count_by_frame = {}
    active_tracks_by_frame = defaultdict(set)
    tracks_to_frames = defaultdict(list)
    frames = []

    for fname in files:
        frame_idx = frame_index_from_name(fname)
        if frame_idx < 0:
            continue
        frames.append(frame_idx)

        path = os.path.join(labels_dir, fname)
        dets_this_frame = 0

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()

                # Expected MOT format:
                # cls track_id x y w h conf
                if len(parts) >= 7:
                    dets_this_frame += 1
                    try:
                        track_id = int(float(parts[1]))
                    except ValueError:
                        continue

                    # Ignore invalid IDs like -1 (they are "untracked" boxes)
                    if track_id < 0:
                        continue

                    active_tracks_by_frame[frame_idx].add(track_id)
                    tracks_to_frames[track_id].append(frame_idx)

                # Detection-only (should not happen now, but safe)
                elif len(parts) == 6:
                    dets_this_frame += 1

        det_count_by_frame[frame_idx] = dets_this_frame

    frames_sorted = sorted(set(frames))
    return frames_sorted, det_count_by_frame, tracks_to_frames, active_tracks_by_frame


def compute_track_stats(frames_sorted, det_count_by_frame, tracks_to_frames, active_tracks_by_frame):
    total_frames = len(frames_sorted)
    dets_per_frame = [det_count_by_frame.get(fi, 0) for fi in frames_sorted]

    avg_dets_per_frame = mean(dets_per_frame)
    med_dets_per_frame = median(dets_per_frame)
    zero_det_ratio = sum(1 for d in dets_per_frame if d == 0) / total_frames

    unique_tracks = len(tracks_to_frames)

    active_tracks_counts = [len(active_tracks_by_frame.get(fi, set())) for fi in frames_sorted]
    avg_active_tracks = mean(active_tracks_counts)
    med_active_tracks = median(active_tracks_counts)

    lifetimes_frames = []
    gaps_per_track = []
    total_gaps = 0

    for tid, frs in tracks_to_frames.items():
        frs_sorted = sorted(set(frs))
        lifetimes_frames.append(len(frs_sorted))

        gaps = 0
        for a, b in zip(frs_sorted, frs_sorted[1:]):
            if b > a + 1:
                gaps += 1
        gaps_per_track.append(gaps)
        total_gaps += gaps

    if lifetimes_frames:
        lifetimes_sec = [lf / FPS for lf in lifetimes_frames]
        mean_life_s = mean(lifetimes_sec)
        med_life_s = median(lifetimes_sec)

        churn_lt_05s = sum(1 for lf in lifetimes_sec if lf < 0.5) / len(lifetimes_sec)
        churn_lt_10s = sum(1 for lf in lifetimes_sec if lf < 1.0) / len(lifetimes_sec)

        lifetimes_sec_sorted = sorted(lifetimes_sec)
        p90_idx = int(round(0.9 * (len(lifetimes_sec_sorted) - 1)))
        p90_life_s = lifetimes_sec_sorted[p90_idx]
    else:
        mean_life_s = med_life_s = p90_life_s = 0.0
        churn_lt_05s = churn_lt_10s = 0.0

    avg_gaps_per_track = mean(gaps_per_track) if gaps_per_track else 0.0

    return {
        "frames": total_frames,
        "avg_dets_per_frame": avg_dets_per_frame,
        "med_dets_per_frame": med_dets_per_frame,
        "zero_det_ratio": zero_det_ratio,
        "unique_tracks": unique_tracks,
        "avg_active_tracks": avg_active_tracks,
        "med_active_tracks": med_active_tracks,
        "mean_track_life_s": mean_life_s,
        "median_track_life_s": med_life_s,
        "p90_track_life_s": p90_life_s,
        "churn_lt_0_5s": churn_lt_05s,
        "churn_lt_1_0s": churn_lt_10s,
        "avg_gaps_per_track": avg_gaps_per_track,
        "total_gaps": total_gaps,
    }, lifetimes_frames


def save_summary_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_lifetime_hist(all_lifetimes_by_run, out_path):
    plt.figure()
    any_series = False
    for run_name, lifetimes_frames in all_lifetimes_by_run.items():
        lifetimes_sec = [lf / FPS for lf in lifetimes_frames]
        if lifetimes_sec:
            plt.hist(lifetimes_sec, bins=40, alpha=0.5, label=run_name)
            any_series = True
    plt.xlabel("Track lifetime (s)")
    plt.ylabel("Count")
    if any_series:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(summary_rows, key, out_path, ylabel):
    plt.figure()
    names = [r["run"] for r in summary_rows]
    vals = [float(r[key]) for r in summary_rows]
    plt.bar(names, vals)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    summary_rows = []
    all_lifetimes_by_run = {}

    for run_name, labels_dir in RUNS.items():
        frames_sorted, det_count_by_frame, tracks_to_frames, active_tracks_by_frame = parse_labels_folder(labels_dir)
        stats, lifetimes_frames = compute_track_stats(frames_sorted, det_count_by_frame, tracks_to_frames, active_tracks_by_frame)

        row = {"run": run_name}
        row.update(stats)
        summary_rows.append(row)
        all_lifetimes_by_run[run_name] = lifetimes_frames

        print(f"\n[{run_name}]")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    csv_path = os.path.join(REPORT_DIR, "tracking_summary.csv")
    save_summary_csv(summary_rows, csv_path)

    plot_lifetime_hist(all_lifetimes_by_run, os.path.join(REPORT_DIR, "lifetime_hist.png"))
    plot_bar(summary_rows, "avg_active_tracks", os.path.join(REPORT_DIR, "avg_active_tracks.png"), "Avg active tracks per frame")
    plot_bar(summary_rows, "churn_lt_1_0s", os.path.join(REPORT_DIR, "churn_lt_1_0s.png"), "Churn: fraction of tracks < 1.0s")
    plot_bar(summary_rows, "avg_dets_per_frame", os.path.join(REPORT_DIR, "avg_dets_per_frame.png"), "Avg detections per frame")

    print(f"\nSaved report to:\n  {REPORT_DIR}")
    print(f"CSV:\n  {csv_path}")


if __name__ == "__main__":
    main()
