"""
Patch existing SAM2 v3 CSVs to:
1. Null out velocity at detection-method transitions (cross-method coord jumps).
2. Null out cx/cy/area/velocity for hough/template rows that fall within the
   spreading phase (impact_frame <= frame <= liftoff_frame) — these are
   false detections on frame artifacts, not the droplet.

Does NOT re-run SAM2. Reads each CSV, patches in-place, writes back.
Idempotent — safe to run multiple times.
"""
import csv
import io
import json
import math
from pathlib import Path

FEATURE_JSON = Path("/home/ubuntu/materials/feature_table.json")
FPS_ACTUAL   = 2996.766489

FOLDER_MAP = {
    "02182026": Path("/home/ubuntu/materials/results_drops/02182026_sam2_v3_results"),
    "03242026": Path("/home/ubuntu/materials/results_drops/03242026_sam2_v3_results"),
    "05052026": Path("/home/ubuntu/materials/results_drops/05052026_sam2_v3_results"),
    "05112026": Path("/home/ubuntu/materials/results_drops/05112026_sam2_v3_results"),
    "05122026": Path("/home/ubuntu/materials/results_drops/05122026_sam2_v3_results"),
    "05172026": Path("/home/ubuntu/materials/results_drops/05172026_sam2_v3_results"),
}

FIELDNAMES = [
    'frame', 'phase', 'drop_id', 'cx', 'cy', 'area_px', 'percentage',
    'detection_method', 'distance_px', 'velocity_px_per_s', 'velocity_mm_s',
]


def null_row_detection(row):
    row['cx']                = ''
    row['cy']                = ''
    row['area_px']           = ''
    row['percentage']        = ''
    row['detection_method']  = 'null'
    row['distance_px']       = ''
    row['velocity_px_per_s'] = ''
    row['velocity_mm_s']     = ''


def patch_csv(csv_path, impact_frame, liftoff_frame, px_per_mm):
    # Infer frame_step from consecutive frame numbers in the CSV
    text = csv_path.read_text()
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return 0

    frames = [int(r['frame']) for r in rows if r.get('frame')]
    frame_step = 1
    if len(frames) >= 2:
        gaps = [frames[i+1] - frames[i] for i in range(min(10, len(frames)-1)) if frames[i+1] > frames[i]]
        if gaps:
            from collections import Counter
            frame_step = Counter(gaps).most_common(1)[0][0]

    # 12 px/frame × frame_step — same formula as analyze_droplet_sam2.py
    max_disp = 12.0 * frame_step

    patched = 0

    # Step 1a: null hough/template detections inside spreading phase
    for row in rows:
        fi = int(row['frame'])
        method = row.get('detection_method', '')
        in_spreading = (impact_frame is not None and liftoff_frame is not None
                        and impact_frame <= fi <= liftoff_frame)
        if in_spreading and method in ('hough', 'template'):
            null_row_detection(row)
            patched += 1

    # Step 1b: spatial consistency check — null hough/template rows that are
    # too far from the last-known position (catches artifact detections in the
    # falling phase for long videos with frame_step=4).
    # Process rows in frame order; track last valid position per drop_id.
    last_known = {}  # drop_id -> (cx, cy)
    for row in rows:
        did    = int(row.get('drop_id') or 0)
        cx_s   = row.get('cx', '')
        cy_s   = row.get('cy', '')
        method = row.get('detection_method', '')
        if not cx_s or not cy_s or did == 0:
            continue
        cx = float(cx_s)
        cy = float(cy_s)
        if method in ('hough', 'template') and did in last_known:
            prev_cx, prev_cy = last_known[did]
            disp = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if disp > max_disp:
                null_row_detection(row)
                patched += 1
                continue  # don't update last_known with this bad detection
        # Update last_known only for valid detections
        if row.get('detection_method', '') not in ('', 'null'):
            last_known[did] = (cx, cy)

    # Step 2: recompute velocity — null across method changes
    last_pos = {}  # drop_id -> (frame, cx, cy, method)
    for row in rows:
        did    = int(row.get('drop_id') or 0)
        cx_s   = row.get('cx', '')
        cy_s   = row.get('cy', '')
        fi     = int(row['frame'])
        method = row.get('detection_method', '')

        row['distance_px']       = ''
        row['velocity_px_per_s'] = ''
        row['velocity_mm_s']     = ''

        if cx_s and cy_s and did != 0:
            cx = float(cx_s)
            cy = float(cy_s)
            if did in last_pos:
                prev_fi, prev_cx, prev_cy, prev_method = last_pos[did]
                frame_gap = fi - prev_fi
                if frame_gap > 0 and method == prev_method and method != 'null':
                    dist   = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    time_s = frame_gap / FPS_ACTUAL
                    vel_px = dist / time_s
                    row['distance_px']       = round(dist, 3)
                    row['velocity_px_per_s'] = round(vel_px, 3)
                    row['velocity_mm_s']     = round(vel_px / px_per_mm, 3)
            last_pos[did] = (fi, cx, cy, method)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(out.getvalue())
    return patched


def main():
    features = json.loads(FEATURE_JSON.read_text())
    lookup = {}
    for e in features:
        stem = Path(e['video']).stem
        lookup[(e['folder'], stem)] = (
            e.get('impact_frame'),
            e.get('liftoff_frame'),
            e.get('px_per_mm', 65.625),
        )

    # px_per_mm not in feature_table — use folder defaults
    folder_px = {
        "02182026": 65.625, "03242026": 65.625,
        "05052026": 66.0,   "05112026": 66.5,
        "05122026": 56.0,   "05172026": 56.0,
    }

    total_files = 0
    total_patched_rows = 0
    for folder_key, out_dir in FOLDER_MAP.items():
        if not out_dir.exists():
            continue
        csvs = sorted(out_dir.glob("*_sam2.csv"))
        px_per_mm = folder_px[folder_key]
        print(f"\n=== {folder_key} ({len(csvs)} CSVs, px/mm={px_per_mm}) ===")
        for csv_path in csvs:
            stem = csv_path.stem
            if stem.endswith('_sam2'):
                stem = stem[:-5]
            impact, liftoff, _ = lookup.get((folder_key, stem), (None, None, None))
            n = patch_csv(csv_path, impact, liftoff, px_per_mm)
            total_files += 1
            if n:
                total_patched_rows += n
                print(f"  [patched] {csv_path.name}  nulled {n} bad spreading rows")
            else:
                print(f"  [ok]      {csv_path.name}")

    print(f"\nDone. {total_files} files processed, {total_patched_rows} rows patched.")


if __name__ == "__main__":
    main()
