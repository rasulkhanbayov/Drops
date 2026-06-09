"""
Compare SAM2 vs OpenCV droplet tracking results for all videos in a folder.

For each video, loads *_sam2.csv and *_results.csv, computes:
  - reference frame & area for each method
  - total rows tracked
  - frames in common between both methods
  - mean centroid distance on common frames (drop_id=1 only)
  - mean |area difference| on common frames (px² and % of reference)

Prints a per-video table and writes compare_summary.csv to the folder.
"""

import os
import csv
import sys
from pathlib import Path
import numpy as np


def load_csv(path):
    """Return list of dicts from CSV, or None if file missing / empty."""
    if not os.path.isfile(path):
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows if rows else None


def reference_row(rows):
    """First row (lowest frame number, drop_id=1)."""
    r1 = [r for r in rows if int(r['drop_id']) == 1]
    if not r1:
        return None
    return min(r1, key=lambda r: int(r['frame']))


def summarise(rows):
    """Return dict: ref_frame, ref_area, total_rows, max_frame."""
    ref = reference_row(rows)
    if ref is None:
        return None
    return {
        'ref_frame': int(ref['frame']),
        'ref_area':  float(ref['area_px']),
        'total_rows': len(rows),
        'max_frame':  max(int(r['frame']) for r in rows),
    }


def compare_on_common_frames(sam2_rows, cv_rows, frame_tol=2):
    """
    Match frames between SAM2 and OpenCV (drop_id=1 only).
    frame_tol handles SAM2 frame_step rounding (e.g. step=4 → nearest ±2).
    Returns (n_common, mean_centroid_dist, mean_area_diff_px, mean_area_diff_pct).
    """
    # Build OpenCV frame→(cx,cy,area) map for drop_id=1
    cv_map = {}
    for r in cv_rows:
        if int(r['drop_id']) == 1:
            cv_map[int(r['frame'])] = (float(r['cx']), float(r['cy']), float(r['area_px']))

    cv_frames_sorted = sorted(cv_map.keys())

    centroid_dists = []
    area_diffs     = []

    for r in sam2_rows:
        if int(r['drop_id']) != 1:
            continue
        sf = int(r['frame'])
        # Find closest CV frame within tolerance
        best = min(cv_frames_sorted, key=lambda f: abs(f - sf), default=None)
        if best is None or abs(best - sf) > frame_tol:
            continue
        s_cx, s_cy, s_area = float(r['cx']), float(r['cy']), float(r['area_px'])
        c_cx, c_cy, c_area = cv_map[best]
        centroid_dists.append(((s_cx - c_cx)**2 + (s_cy - c_cy)**2)**0.5)
        area_diffs.append(abs(s_area - c_area))

    if not centroid_dists:
        return 0, None, None, None

    # Use OpenCV reference area for % calculation
    ref_r = reference_row(cv_rows)
    ref_area = float(ref_r['area_px']) if ref_r else 1.0

    return (
        len(centroid_dists),
        float(np.mean(centroid_dists)),
        float(np.mean(area_diffs)),
        float(np.mean(area_diffs) / ref_area * 100),
    )


def main(folder: str):
    folder = Path(folder)
    videos = sorted(folder.glob('*.mp4'))

    summary_rows = []

    print(f"\n{'Video':<28} {'Method':<8} {'Ref Fr':>7} {'Ref Area':>10} {'Rows':>6} {'Max Fr':>7}")
    print('-' * 75)

    for v in videos:
        stem = v.stem
        sam2_path = folder / f"{stem}_sam2.csv"
        cv_path   = folder / f"{stem}_results.csv"

        sam2 = load_csv(str(sam2_path))
        cv   = load_csv(str(cv_path))

        s_sum = summarise(sam2) if sam2 else None
        c_sum = summarise(cv)   if cv   else None

        name = stem[:28]
        if s_sum:
            print(f"{name:<28} {'SAM2':<8} {s_sum['ref_frame']:>7} {s_sum['ref_area']:>10.0f} "
                  f"{s_sum['total_rows']:>6} {s_sum['max_frame']:>7}")
        else:
            print(f"{name:<28} {'SAM2':<8} {'N/A':>7} {'N/A':>10} {'N/A':>6} {'N/A':>7}")

        if c_sum:
            print(f"{'':28} {'OpenCV':<8} {c_sum['ref_frame']:>7} {c_sum['ref_area']:>10.0f} "
                  f"{c_sum['total_rows']:>6} {c_sum['max_frame']:>7}")
        else:
            print(f"{'':28} {'OpenCV':<8} {'N/A':>7} {'N/A':>10} {'N/A':>6} {'N/A':>7}")

        # Per-frame comparison
        n_common = mean_cd = mean_ad = mean_ad_pct = None
        if sam2 and cv:
            n_common, mean_cd, mean_ad, mean_ad_pct = compare_on_common_frames(sam2, cv)
            if n_common:
                print(f"{'':28} {'DIFF':<8} "
                      f"{'common frames:':>18} {n_common:>5}  "
                      f"centroid dist: {mean_cd:>6.1f} px  "
                      f"area diff: {mean_ad:>7.0f} px² ({mean_ad_pct:>5.1f}%)")
        print()

        summary_rows.append({
            'video':              stem,
            'sam2_ref_frame':     s_sum['ref_frame']  if s_sum else '',
            'sam2_ref_area':      s_sum['ref_area']   if s_sum else '',
            'sam2_total_rows':    s_sum['total_rows'] if s_sum else '',
            'sam2_max_frame':     s_sum['max_frame']  if s_sum else '',
            'cv_ref_frame':       c_sum['ref_frame']  if c_sum else '',
            'cv_ref_area':        c_sum['ref_area']   if c_sum else '',
            'cv_total_rows':      c_sum['total_rows'] if c_sum else '',
            'cv_max_frame':       c_sum['max_frame']  if c_sum else '',
            'common_frames':      n_common if n_common is not None else '',
            'mean_centroid_dist_px': f"{mean_cd:.2f}" if mean_cd is not None else '',
            'mean_area_diff_px':     f"{mean_ad:.0f}" if mean_ad is not None else '',
            'mean_area_diff_pct':    f"{mean_ad_pct:.2f}" if mean_ad_pct is not None else '',
        })

    out_csv = folder / 'compare_summary.csv'
    fields = [
        'video',
        'sam2_ref_frame', 'sam2_ref_area', 'sam2_total_rows', 'sam2_max_frame',
        'cv_ref_frame',   'cv_ref_area',   'cv_total_rows',   'cv_max_frame',
        'common_frames', 'mean_centroid_dist_px', 'mean_area_diff_px', 'mean_area_diff_pct',
    ]
    with open(str(out_csv), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Summary written → {out_csv}")


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '05052026'
    main(folder)
