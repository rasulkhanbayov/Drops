"""
Extract Key Features — All Videos
==================================
For every video in both dataset folders, extract the core physical
measurements using classical computer vision:

  • impact_frame, liftoff_frame
  • contact_time_ms        — duration droplet touches surface
  • pre_impact_diameter_mm — droplet size before impact
  • pre_impact_radius_px
  • pre_impact_cx, pre_impact_cy  — centroid just before impact
  • max_spread_width_px / _mm     — peak horizontal contact footprint
  • max_spread_factor             — β_max = D_spread / D_0
  • impact_velocity_px_per_frame  — estimated from centroid drop in falling phase
  • impact_velocity_mm_per_s      — converted to physical units

Output
------
  feature_table.csv   — one row per video, all features
  feature_table.json  — same data as JSON

Usage
-----
    /opt/anaconda3/2024.02-1/conda_envs/ml_dl_gpu_base/bin/python \
        extract_features.py
"""

import cv2
import csv
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
PX_PER_MM   = 65.625
FPS_ACTUAL  = 2996.766489   # true capture rate at 1280×512 (from CameraSpecs.jpg)
FPS_ENCODED = 60.0          # video file playback FPS

VIDEOS_02 = Path("/home/ubuntu/materials/02182026")
VIDEOS_03 = Path("/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant")
OUT_DIR   = Path("/home/ubuntu/materials")

SURFACE_ROW_02: dict[str, int] = {
    "water.mp4":     433, "water2.mp4":  433, "water3.mp4":  433,
    "water4.mp4":    417, "water5.mp4":  417, "water6.mp4":  426,
    "cainhcg1.mp4":  400, "cainhcg2.mp4":433, "cainhcg3.mp4":437,
    "cainhcg4.mp4":  433, "cainhcg5.mp4":433,
    "cainhsds1.mp4": 433, "cainhsds2.mp4":430, "cainhsds3.mp4":428,
    "cainhtx1.mp4":  428, "cainhtx2.mp4":428, "cainhtx3.mp4":402,
    "cainlcg1.mp4":  433, "cainlcg2.mp4":433, "cainlcg3.mp4":399,
    "cainlsds1.mp4": 427, "cainlsds2.mp4":426, "cainlsds3.mp4":417,
    "cainltx1.mp4":  433, "cainltx2.mp4":428, "cainltx3.mp4":422,
    "caonly1.mp4":   399, "caonly2.mp4":405,  "caonly3.mp4":433,
    "tx.mp4":        417,
}

SURFACE_ROW_03: dict[str, int] = {
    "0.001percent cg.mp4":       404,
    "0.028p.mp4":                404,
    "0.028percrnt tx.mp4":       467,
    "0.45percrnt sds.mp4":       454,
    "ONLY CA SDS ABOVE CMC.mp4": 481, "ONLY CA SDS ABOVE CMC1.mp4": 481,
    "ONLY CA SDS ABOVE CMC2.mp4":481,
    "ONLY CA cg ABOVE CMC1.mp4": 485, "ONLY CA cg ABOVE CMC2.mp4":  481,
    "ONLY CA cg ABOVE CMC3.mp4": 473,
    "ONLY CA cg less CMC1.mp4":  470, "ONLY CA cg less CMC2.mp4":   465,
    "ONLY CA cg less CMC3.mp4":  473,
    "ONLY CA sds less CMC1.mp4": 471, "ONLY CA sds less CMC2.mp4":  470,
    "ONLY CA tx ABOVE CMC1.mp4": 482, "ONLY CA tx ABOVE CMC2.mp4":  471,
    "ONLY CA tx ABOVE CMC3.mp4": 470, "ONLY CA tx ABOVE CMC4.mp4":  471,
    "ONLY CA tx less CMC1.mp4":  465,
    "ONLY CA tx less CMC2.mp4":  503, "ONLY CA tx less CMC3.mp4":   505,
    "ca+TR.mp4":                 479,
}


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class VideoFeatures:
    folder:                    str
    video:                     str
    surface_row_px:            int
    impact_frame:              int
    liftoff_frame:             int
    contact_time_ms:           float          # (liftoff - impact) / FPS * 1000
    pre_impact_cx_px:          Optional[float]
    pre_impact_cy_px:          Optional[float]
    pre_impact_radius_px:      Optional[float]
    pre_impact_diameter_mm:    Optional[float]
    max_spread_width_px:       Optional[float]
    max_spread_width_mm:       Optional[float]
    max_spread_factor:         Optional[float]  # β_max = D_spread / D_0
    impact_velocity_px_per_frame: Optional[float]
    impact_velocity_mm_per_s:  Optional[float]


# ── CV helpers ────────────────────────────────────────────────────────────────
def read_frame(path: str, fi: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def hough_detect(gray: np.ndarray, min_r=20, max_r=100) -> Optional[tuple]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=30, param1=50, param2=20,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    c = np.round(circles[0]).astype(int)
    best = sorted(c, key=lambda x: x[1])[0]   # uppermost circle
    return float(best[0]), float(best[1]), float(best[2])


def contact_width(gray: np.ndarray, surface_y: int) -> Optional[float]:
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    band = thresh[max(0, surface_y - 10): surface_y + 5, :]
    cols = np.where(band.max(axis=0) > 0)[0]
    return float(cols[-1] - cols[0]) if len(cols) >= 5 else None


def find_impact_frame(path: str) -> int:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev, diffs = None, []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            roi = gray[350:, :]
            diffs.append((i, float(np.abs(roi.astype(int) - prev[350:, :].astype(int)).mean())))
        prev = gray
    cap.release()
    return max(diffs, key=lambda x: x[1])[0]


def find_liftoff_frame(path: str, impact: int, surface_y: int, window=300) -> int:
    """
    Detect liftoff by finding the first frame (after impact+10) where
    HoughCircles successfully detects a circle that is fully airborne —
    i.e. its centroid is at least (radius + 5 px) above the surface row.

    During spreading the droplet is flat; HoughCircles won't find a clean
    circle.  After liftoff it finds a rising sphere, which is the signal
    we want.  Falls back to impact+window if nothing is found.
    """
    cap = cv2.VideoCapture(path)
    liftoff = impact + window
    for i in range(impact + 10, impact + window):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = hough_detect(gray, min_r=15, max_r=120)
        if det is not None:
            cx, cy, r = det
            # Circle must be fully above the surface (centroid + radius < surface_y)
            if cy + r < surface_y - 5:
                liftoff = i
                break
    cap.release()
    return liftoff


def measure_pre_impact(path: str, impact: int) -> tuple:
    """
    Sample up to 5 frames before impact, return median cx, cy, radius.
    Also estimate vertical velocity from centroid displacement.
    """
    centroid_ys = []
    detections  = []
    for offset in range(8, 2, -1):   # frames: impact-8 .. impact-3
        fi = impact - offset
        if fi < 0:
            continue
        frame = read_frame(path, fi)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = hough_detect(gray)
        if det:
            detections.append(det)
            centroid_ys.append((fi, det[1]))

    if not detections:
        return None, None, None, None

    cx     = float(np.median([d[0] for d in detections]))
    cy     = float(np.median([d[1] for d in detections]))
    radius = float(np.median([d[2] for d in detections]))

    # Velocity: linear fit of cy vs frame_index (cy increases as droplet falls)
    velocity = None
    if len(centroid_ys) >= 2:
        frames_arr = np.array([p[0] for p in centroid_ys], dtype=float)
        ys_arr     = np.array([p[1] for p in centroid_ys], dtype=float)
        coeffs     = np.polyfit(frames_arr, ys_arr, 1)
        velocity   = float(coeffs[0])   # px per encoded frame

    return cx, cy, radius, velocity


def measure_max_spread(path: str, impact: int, liftoff: int, surface_y: int) -> Optional[float]:
    """Scan the spreading phase and return the maximum contact width."""
    max_w = None
    for fi in range(impact + 1, liftoff):
        frame = read_frame(path, fi)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w = contact_width(gray, surface_y)
        if w is not None:
            max_w = w if max_w is None else max(max_w, w)
    return max_w


# ── Per-video extraction ───────────────────────────────────────────────────────
def process_video(video_path: Path, surface_y: int, folder_label: str) -> VideoFeatures:
    path = str(video_path)

    impact  = find_impact_frame(path)
    liftoff = find_liftoff_frame(path, impact, surface_y)

    contact_ms = round((liftoff - impact) / FPS_ACTUAL * 1000, 4)

    cx, cy, radius, vel_px_per_enc_frame = measure_pre_impact(path, impact)

    # Convert per-frame velocity to physical mm/s.
    # The video stores every actual captured frame, so each frame index step
    # = 1/FPS_ACTUAL seconds of real time.
    #   velocity (px/s) = vel_px_per_frame × FPS_ACTUAL
    #   velocity (mm/s) = velocity (px/s) / PX_PER_MM
    vel_mm_s = None
    if vel_px_per_enc_frame is not None:
        vel_px_per_s = vel_px_per_enc_frame * FPS_ACTUAL
        vel_mm_s = round(abs(vel_px_per_s) / PX_PER_MM, 2)
        vel_px_per_enc_frame = round(vel_px_per_enc_frame, 4)

    diameter_mm = round(2 * radius / PX_PER_MM, 4) if radius is not None else None

    max_sw_px = measure_max_spread(path, impact, liftoff, surface_y)
    max_sw_mm = round(max_sw_px / PX_PER_MM, 4) if max_sw_px is not None else None

    beta_max = None
    if max_sw_px is not None and radius is not None and radius > 0:
        D0 = 2 * radius
        beta_max = round(max_sw_px / D0, 4)

    return VideoFeatures(
        folder=folder_label,
        video=video_path.name,
        surface_row_px=surface_y,
        impact_frame=impact,
        liftoff_frame=liftoff,
        contact_time_ms=contact_ms,
        pre_impact_cx_px=round(cx, 1) if cx is not None else None,
        pre_impact_cy_px=round(cy, 1) if cy is not None else None,
        pre_impact_radius_px=round(radius, 1) if radius is not None else None,
        pre_impact_diameter_mm=diameter_mm,
        max_spread_width_px=round(max_sw_px, 1) if max_sw_px is not None else None,
        max_spread_width_mm=max_sw_mm,
        max_spread_factor=beta_max,
        impact_velocity_px_per_frame=vel_px_per_enc_frame,
        impact_velocity_mm_per_s=vel_mm_s,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    all_features: list[VideoFeatures] = []

    folders = [
        (VIDEOS_02, SURFACE_ROW_02, "02182026"),
        (VIDEOS_03, SURFACE_ROW_03, "03242026"),
    ]

    for videos_dir, surface_map, folder_label in folders:
        if not videos_dir.exists():
            print(f"  [skip] {videos_dir} not found")
            continue

        video_files = sorted(videos_dir.glob("*.mp4"))
        print(f"\n── {folder_label}  ({len(video_files)} videos) ──────────────────────")

        for vf in video_files:
            surface_y = surface_map.get(vf.name)
            if surface_y is None:
                print(f"  [skip] {vf.name}  — no surface row defined")
                continue

            print(f"  {vf.name:<42}", end="", flush=True)
            try:
                feat = process_video(vf, surface_y, folder_label)
                all_features.append(feat)
                print(
                    f"impact={feat.impact_frame:5d}  "
                    f"contact={feat.contact_time_ms:.2f}ms  "
                    f"D0={feat.pre_impact_diameter_mm or '?':>6}mm  "
                    f"β_max={feat.max_spread_factor or '?':>6}"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "feature_table.csv"
    col_names = [f.name for f in fields(VideoFeatures)]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=col_names)
        writer.writeheader()
        for feat in all_features:
            writer.writerow(asdict(feat))

    # ── Write JSON ─────────────────────────────────────────────────────────────
    json_path = OUT_DIR / "feature_table.json"
    json_path.write_text(json.dumps(
        [asdict(f) for f in all_features], indent=2
    ))

    print(f"\nDone.  {len(all_features)} videos processed.")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")


if __name__ == "__main__":
    main()
