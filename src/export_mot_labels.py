import os
import re
from ultralytics import YOLO

# ====== PATHS ======
FRAMES_DIR = r"C:\Users\dhruv\carla_sim\frames"
OUT_BASE = r"C:\Users\dhruv\carla_sim\outputs"
MODEL_PATH = "yolov8s.pt"

# ====== SETTINGS ======
DEVICE = "cpu"   # my torch is CPU only
IMGSZ = 640      
BATCH = 1        # safest for CPU + Windows

# ====== EXPERIMENT GRID ======
CONFIGS = [
    ("bt_conf40_mot", 0.40, "bytetrack.yaml"),
]


def frame_idx_from_path(p: str) -> int:
    """Extract integer from filename like frame_000123.jpg -> 123."""
    name = os.path.basename(p)
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    if not os.path.isdir(FRAMES_DIR):
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")
    os.makedirs(OUT_BASE, exist_ok=True)

    model = YOLO(MODEL_PATH)

    for run_name, conf, tracker in CONFIGS:
        run_dir = os.path.join(OUT_BASE, run_name)
        labels_dir = os.path.join(run_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        # IMPORTANT: persist=True -> tracker carries state across frames -> real IDs
        results = model.track(
            source=FRAMES_DIR,
            conf=conf,
            tracker=tracker,
            persist=True,
            stream=True,
            save=False,       # set True if you want visuals; not needed for metrics
            verbose=False,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            vid_stride=1,     # process every frame
        )

        n_frames = 0
        nonneg_id_count = 0
        total_boxes = 0

        for r in results:
            img_path = getattr(r, "path", "")
            fi = frame_idx_from_path(img_path)
            if fi < 0:
                fi = n_frames

            out_path = os.path.join(labels_dir, f"frame_{fi:06d}.txt")

            lines = []
            if r.boxes is not None and len(r.boxes) > 0:
                xywhn = r.boxes.xywhn.cpu().numpy()     # normalized xywh
                cls = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                ids = r.boxes.id
                if ids is not None:
                    ids = ids.cpu().numpy().astype(int)
                else:
                    ids = [-1] * len(cls)

                for j in range(len(cls)):
                    x, y, w, h = xywhn[j]
                    tid = int(ids[j])
                    total_boxes += 1
                    if tid >= 0:
                        nonneg_id_count += 1
                    lines.append(
                        f"{cls[j]} {tid} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {confs[j]:.6f}"
                    )

            # Always write a file to get exactly 4500 .txt files (aligns with raw 7m30s clip)
            with open(out_path, "w") as f:
                f.write("\n".join(lines))

            n_frames += 1

        print(f"[OK] {run_name}: wrote {n_frames} label files -> {labels_dir}")
        print(f"     boxes written: {total_boxes}, boxes with valid IDs (>=0): {nonneg_id_count}")

if __name__ == "__main__":
    main()
