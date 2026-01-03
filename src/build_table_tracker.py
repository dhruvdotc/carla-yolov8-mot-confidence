# reads all per-frame MOT label .txt files
# Merge them into ONE clean CSV table for pandas style analysis

import os
import csv
import re

# =========================
# USER PATHS
# =========================

LABELS_DIR = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/labels"
OUTPUT_CSV = "C:/Users/dhruv/carla_sim/outputs/bs_conf50_mot/tracking.csv"

MIN_CONFIDENCE = 0.50
DROP_UNTRACKED = True   # drop track_id = -1

# =========================
# FUNCTION DEFINITIONS
# =========================

def get_frame_number(filename):
    # Extract the frame index from a filename by taking the first number that appears in the filename
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    return -1


def read_label_file(path):
    # reads ONE mot-style label file (one frame) and return all detections inside it
    # note:
    # x, y are already center coordinates (YOLO xywh format)

    rows = []
    file = open(path, "r")

    for line in file:
        line = line.strip()
        if line == "":
            continue

        parts = line.split()

        if len(parts) >= 7:
            cls  = int(float(parts[0]))
            tid  = int(float(parts[1]))
            cx   = float(parts[2])
            cy   = float(parts[3])
            w    = float(parts[4])
            h    = float(parts[5])
            conf = float(parts[6])
            rows.append((cls, tid, cx, cy, w, h, conf))


    file.close()
    return rows


# =========================
# MAIN SCRIPT LOGIC
# =========================

# List all label files
files = os.listdir(LABELS_DIR)
files.sort()

# Open CSV for writing
csv_file = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csv_file)

# CSV header (IBM / pandas friendly)
writer.writerow(["frame", "track_id", "class", "cx", "cy", "w", "h", "confidence"])

total_read = 0
total_written = 0

# Loop through every frame file
for filename in files:
    if not filename.endswith(".txt"):
        continue

    frame = get_frame_number(filename)
    if frame < 0:
        continue

    full_path = os.path.join(LABELS_DIR, filename)
    detections = read_label_file(full_path)

    # Write each valid detection as one row
    for cls, tid, cx, cy, w, h, conf in detections:
        total_read += 1

        # Confidence filtering
        if conf < MIN_CONFIDENCE:
            continue

        # Drop detections without stable tracking IDs
        if DROP_UNTRACKED and tid < 0:
            continue

        writer.writerow([frame, tid, cls, cx, cy, w, h, conf])
        total_written += 1

csv_file.close()

print("DONE")
print("Input label files:", len(files))
print("Rows read:", total_read)
print("Rows written:", total_written)
print("Saved to:", OUTPUT_CSV)