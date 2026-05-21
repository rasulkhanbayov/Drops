"""
Recover null D0 / U0 values in feature_table.json
===================================================
For every video that still has null pre_impact_* or impact_velocity,
re-attempts detection with:
  1. Wider frame lookback (up to 40 frames before impact)
  2. Relaxed HoughCircles params (lower param2, wider radius range)
  3. Contour-based fallback (largest dark blob above surface)
  4. Velocity from the best-fit slope across all valid detections

Updates feature_table.json in-place and saves the changed entries
to feature_table_recovered.json.
"""

import cv2
import json
import numpy as np
from pathlib import Path

PX_PER_MM   = 65.625
FPS_ACTUAL  = 2996.766489

VIDEOS_02 = Path("/home/ubuntu/materials/02182026")
VIDEOS_03 = Path("/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant")

FEATURE_JSON = Path("/home/ubuntu/materials/feature_table.json")
OUT_JSON     = Path("/home/ubuntu/materials/feature_table.json")         # update in-place
DIFF_JSON    = Path("/home/ubuntu/materials/feature_table_recovered.json")


def read_frame(path: str, fi: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def hough_detect_multi(gray, min_r=15, max_r=140, param2=18):
    """Try multiple param2 values; return first success."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for p2 in [param2, 15, 12, 10]:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=30, param1=50, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            # Return uppermost circle (lowest cy = farthest from surface)
            best = sorted(c, key=lambda x: x[1])[0]
            return float(best[0]), float(best[1]), float(best[2])
    return None


def contour_fallback(gray, surface_y):
    """
    Find the largest dark blob strictly above the surface.
    Returns (cx, cy, radius_equiv) or None.
    """
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    # Mask out the surface band and below
    thresh[surface_y - 15:, :] = 0
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Pick largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:   # too small
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    r  = float(np.sqrt(area / np.pi))
    return float(cx), float(cy), r


def measure_pre_impact_robust(video_path: str, impact: int, surface_y: int):
    """
    Extended pre-impact detection:
      - Search up to 40 frames before impact
      - Try Hough then contour fallback per frame
      - Require the detected blob centroid to be above surface_y - radius - 5
      - Fit velocity from all valid detections
    """
    detections = []
    for offset in range(40, 1, -1):
        fi = impact - offset
        if fi < 0:
            continue
        frame = read_frame(video_path, fi)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        det = hough_detect_multi(gray)
        if det is None:
            det = contour_fallback(gray, surface_y)

        if det is not None:
            cx, cy, r = det
            # Sanity: blob must be above surface and not too close to edges
            if cy + r < surface_y - 5 and r > 10 and cx > 10 and cx < 1270:
                detections.append((fi, cx, cy, r))

    if not detections:
        return None, None, None, None

    cx     = float(np.median([d[1] for d in detections]))
    cy     = float(np.median([d[2] for d in detections]))
    radius = float(np.median([d[3] for d in detections]))

    # Velocity from linear fit on cy vs frame_index (needs ≥2 points)
    velocity = None
    if len(detections) >= 2:
        frames_arr = np.array([d[0] for d in detections], dtype=float)
        ys_arr     = np.array([d[2] for d in detections], dtype=float)
        coeffs     = np.polyfit(frames_arr, ys_arr, 1)
        velocity   = float(coeffs[0])   # px per actual frame

    return cx, cy, radius, velocity


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    features = json.loads(FEATURE_JSON.read_text())

    video_dir_map = {
        "02182026": VIDEOS_02,
        "03242026": VIDEOS_03,
    }

    recovered = []
    changed   = 0

    for f in features:
        needs_d0  = f["pre_impact_diameter_mm"] is None
        needs_vel = f["impact_velocity_mm_per_s"] is None or \
                    abs(f.get("impact_velocity_mm_per_s") or 0) < 50  # < 50 mm/s is unrealistic

        if not needs_d0 and not needs_vel:
            continue  # already complete

        video_path = str(video_dir_map[f["folder"]] / f["video"])
        surface_y  = f["surface_row_px"]
        impact     = f["impact_frame"]

        print(f"  Recovering: {f['video']}", end="  ", flush=True)

        cx, cy, radius, vel_px = measure_pre_impact_robust(video_path, impact, surface_y)

        old_d0  = f["pre_impact_diameter_mm"]
        old_vel = f["impact_velocity_mm_per_s"]

        if radius is not None and needs_d0:
            f["pre_impact_cx_px"]       = round(cx, 1)
            f["pre_impact_cy_px"]       = round(cy, 1)
            f["pre_impact_radius_px"]   = round(radius, 1)
            f["pre_impact_diameter_mm"] = round(2 * radius / PX_PER_MM, 4)

        if vel_px is not None and needs_vel:
            vel_mm_s = abs(vel_px) * FPS_ACTUAL / PX_PER_MM
            f["impact_velocity_px_per_frame"] = round(vel_px, 4)
            f["impact_velocity_mm_per_s"]     = round(vel_mm_s, 2)

        new_d0  = f["pre_impact_diameter_mm"]
        new_vel = f["impact_velocity_mm_per_s"]

        status = []
        if old_d0  is None and new_d0  is not None: status.append(f"D0={new_d0}mm")
        if (old_vel is None or abs(old_vel or 0) < 50) and new_vel is not None and new_vel > 50:
            status.append(f"U0={new_vel}mm/s")
        if not status:
            status.append("still null")

        print(", ".join(status))

        if any("null" not in s for s in status):
            changed += 1
            recovered.append({
                "video":  f["video"],
                "folder": f["folder"],
                "pre_impact_diameter_mm":    new_d0,
                "impact_velocity_mm_per_s":  new_vel,
            })

    # Write updated feature_table.json
    OUT_JSON.write_text(json.dumps(features, indent=2))
    DIFF_JSON.write_text(json.dumps(recovered, indent=2))

    print(f"\nUpdated {changed} videos → {OUT_JSON}")
    print(f"Recovery summary → {DIFF_JSON}")

    # Final null count
    still_null = sum(1 for f in features if f["pre_impact_diameter_mm"] is None)
    print(f"Remaining null D0: {still_null} / {len(features)}")


if __name__ == "__main__":
    main()
