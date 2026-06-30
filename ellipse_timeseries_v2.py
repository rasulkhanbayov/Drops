"""
ellipse_timeseries_v2.py
========================
Improved version of ellipse_timeseries.py.  Three targeted upgrades:

  1. TWO-PASS pre-impact scan
       Pass 1  - HoughCircles (broad radius 45-110 px): robust D0 measurement.
                 Same as v1; keeps the +/-1-3% accuracy we already have.
       Pass 2  - Template matching (disk sized to D0): accurate velocity.
                 Template cross-correlation is immune to the nozzle/artifact
                 circles that corrupted v1's HoughCircles-based velocity.

  2. IMPACT-FRAME REFINEMENT
       Scan +/-8 frames around the estimated impact_frame to find the first frame
       where the contact footprint exceeds 1 mm.  Corrects +/-5-frame errors in
       the classical-CV impact detection -> improves beta_max and U0.

  3. OPTICAL FLOW FALLBACK
       If template matching collects < 3 positions (very low-contrast videos),
       Lucas-Kanade optical flow starting from the last HoughCircles detection
       tracks the droplet frame-by-frame.

Outputs to separate paths so v1 is untouched:
  /home/ubuntu/materials/timeseries_v2/<video>_timeseries.csv
  /home/ubuntu/materials/summary_timeseries_v2.json
"""

import argparse
import os
import cv2
import json
import csv
import numpy as np
from pathlib import Path

# -- Constants -----------------------------------------------------------------
FPS_ACTUAL      = 2996.766489
PX_PER_MM       = 65.625
PX_PER_MM_NEW   = 66.0
PX_PER_MM_NEW1  = 66.5   # new_experiments/05112026
PX_PER_MM_NEW2  = 56.0   # new_experiments/05122026

VIDEOS_02   = Path("/home/ubuntu/materials/02182026")
VIDEOS_03   = Path("/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant")
VIDEOS_05   = Path("/home/ubuntu/materials/05052026")
VIDEOS_NEW1 = Path("/home/ubuntu/materials/new_experiments/05112026")
VIDEOS_NEW2 = Path("/home/ubuntu/materials/new_experiments/05122026")

FEATURE_JSON = Path("/home/ubuntu/materials/feature_table.json")
OUT_DIR      = Path("/home/ubuntu/materials/timeseries_v2")
SUMMARY_JSON = Path("/home/ubuntu/materials/summary_timeseries_v2.json")

OUT_DIR.mkdir(exist_ok=True)

CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

CSV_FIELDS = [
    "frame", "phase", "cx_px", "cy_px", "radius_px", "spread_width_px",
    "detection_method", "confidence",
    "D_mm", "beta", "velocity_mm_s", "time_ms",
    "dist_travelled_px", "px_per_mm",
    "state_change",
]


# -- Low-level helpers (unchanged from v1) -------------------------------------

def read_frame(path, fi):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def preprocess(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return CLAHE.apply(gray)


def hough_detect(gray, min_r=15, max_r=140, prefer_largest=False,
                 radius_min_accept=0, radius_max_accept=9999):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for p2 in [20, 15, 12, 10]:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=30, param1=50, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            c = [ci for ci in c if radius_min_accept <= ci[2] <= radius_max_accept]
            if not c:
                continue
            if prefer_largest:
                best = sorted(c, key=lambda x: -x[2])[0]
            else:
                best = sorted(c, key=lambda x: x[1])[0]
            return float(best[0]), float(best[1]), float(best[2])
    return None


def ellipse_params_from_circle(gray, cx, cy, radius):
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 200]
    if valid and len(max(valid, key=cv2.contourArea)) >= 5:
        contour = max(valid, key=cv2.contourArea)
    else:
        contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours2:
            return None
        contour = contours2[0]
    if len(contour) < 5:
        return None
    (ex, ey), (ax1, ax2), angle = cv2.fitEllipse(contour)
    major = max(ax1, ax2)
    minor = min(ax1, ax2)
    area      = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circ      = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    AR        = major / minor if minor > 0 else 1.0
    roundness = (4 * area) / (np.pi * major ** 2) if major > 0 else 1.0
    hull      = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity  = area / hull_area if hull_area > 0 else 1.0
    hull_pts  = hull.reshape(-1, 2).astype(float)
    if len(hull_pts) > 40:
        idx = np.linspace(0, len(hull_pts) - 1, 40, dtype=int)
        hull_pts = hull_pts[idx]
    feret, fx, fy = 0.0, float(ex), float(ey)
    for i in range(len(hull_pts)):
        for j in range(i + 1, len(hull_pts)):
            d = float(np.linalg.norm(hull_pts[i] - hull_pts[j]))
            if d > feret:
                feret, fx, fy = d, hull_pts[i][0], hull_pts[i][1]
    pixels   = gray[mask > 0]
    mean_val = float(pixels.mean()) if len(pixels) else 0.0
    min_val  = int(pixels.min())    if len(pixels) else 0
    max_val  = int(pixels.max())    if len(pixels) else 0
    D_px     = float(np.sqrt(major * minor))
    return dict(
        cx=round(float(ex), 1), cy=round(float(ey), 1),
        area=round(area, 1), mean=round(mean_val, 3),
        min=min_val, max=max_val,
        major=round(float(major), 3), minor=round(float(minor), 3),
        angle=round(float(angle), 1), circ=round(circ, 3),
        feret=round(feret, 1), feret_x=round(fx, 1), feret_y=round(fy, 1),
        min_feret=round(float(minor), 3),
        AR=round(AR, 3), roundness=round(roundness, 3), solidity=round(solidity, 3),
        D_px=round(D_px, 3),
    )


def contact_width_px(frame, background, surface_y):
    diff  = cv2.absdiff(frame, background)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    band  = thresh[max(0, surface_y - 40): surface_y + 10, :]
    cols  = np.where(band.any(axis=0))[0]
    return int(cols[-1] - cols[0]) if len(cols) > 1 else 0


def fill_velocities(rows, positions, fps, px_per_mm):
    for i in range(1, len(positions)):
        dx = positions[i][1] - positions[i - 1][1]
        dy = positions[i][2] - positions[i - 1][2]
        dist_px  = float(np.sqrt(dx ** 2 + dy ** 2))
        vel_mm_s = dist_px * fps / px_per_mm
        rows[i]["dist_travelled_px"] = round(dist_px, 3)
        rows[i]["velocity_mm_s"]     = round(vel_mm_s, 2)


def filter_falling_run(positions):
    """Longest consecutive run with cy strictly increasing and frame gap <= 3."""
    if len(positions) < 2:
        return positions
    best_start, best_len = 0, 1
    curr_start, curr_len = 0, 1
    for i in range(1, len(positions)):
        fi_gap  = positions[i][0] - positions[i - 1][0]
        cy_diff = positions[i][2] - positions[i - 1][2]
        if cy_diff > 0 and fi_gap <= 3:
            curr_len += 1
            if curr_len > best_len:
                best_len  = curr_len
                best_start = curr_start
        else:
            curr_start = i
            curr_len   = 1
    return positions[best_start: best_start + best_len]


def median_pairwise_speed(positions, fps, px_per_mm):
    """
    Theil-Sen velocity: median of adjacent-pair slopes over a monotonic run.
    Returns speed in mm/s, or None if < 2 usable pairs.
    """
    pts = filter_falling_run(positions)
    if len(pts) < 2:
        return None
    fis = np.array([p[0] for p in pts], dtype=float)
    ys  = np.array([p[2] for p in pts], dtype=float)
    if len(pts) >= 3:
        slopes = [
            (ys[i + 1] - ys[i]) / (fis[i + 1] - fis[i])
            for i in range(len(pts) - 1) if fis[i + 1] > fis[i]
        ]
        if slopes:
            slope = float(np.median(slopes))
            spd   = round(abs(slope) * fps / px_per_mm, 2)
            return spd if spd <= 2500 else None
    slope = np.polyfit(fis, ys, 1)[0]
    spd   = round(abs(float(slope)) * fps / px_per_mm, 2)
    return spd if spd <= 2500 else None


# -- NEW: Template matching helpers --------------------------------------------

def make_disk_template(radius_px):
    """
    Shadowgraphy droplet template: dark interior (low-reflectance droplet)
    with a bright caustic ring at the boundary, on a gray background.
    Normalized TM_CCOEFF_NORMED handles absolute brightness variation.
    """
    r  = max(5, int(round(radius_px)))
    sz = 2 * r + 20          # border padding so the ring is fully inside
    tmpl = np.full((sz, sz), 160, dtype=np.uint8)   # neutral gray background
    cx, cy = sz // 2, sz // 2
    cv2.circle(tmpl, (cx, cy), r,     30,  -1)       # dark droplet body
    cv2.circle(tmpl, (cx, cy), r,     230,  3)       # bright caustic ring
    cv2.circle(tmpl, (cx, cy), r - 4, 60,   2)       # inner shadow gradient
    return tmpl


def template_track(video_path, impact_frame, surface_y, radius_px,
                   px_per_mm, lookback=40, conf_thresh=0.30):
    """
    IMPROVEMENT 1 - Pass-2 velocity tracking via normalized template matching.

    For each pre-impact frame (backward scan), slides a droplet-shaped template
    across the search region and records the best-match centre.  Returns a list
    of (frame_idx, cx, cy) positions in chronological order.

    Advantages over HoughCircles for velocity:
      * Searches only where the droplet is expected (above surface, not at top)
      * Template shape is sized to D0, so nozzle/shadow artifacts (different
        size / texture) score far below the confidence threshold
      * Every frame in the scan produces one result; no frame-gap issues
    """
    template      = make_disk_template(radius_px)
    th, tw        = template.shape
    r             = radius_px
    positions     = []
    confidences   = []

    for offset in range(2, lookback + 1):
        fi = impact_frame - offset
        if fi < 0:
            continue
        frame = read_frame(video_path, fi)
        if frame is None:
            continue
        gray = preprocess(frame)
        h, w = gray.shape

        # Search band: between top margin and (surface - radius)
        y_top = max(0, int(r) - 5)
        y_bot = int(surface_y - r - 8)
        if y_bot - y_top < th:
            continue

        search = gray[y_top: y_bot + th, :]
        if search.shape[0] < th or search.shape[1] < tw:
            continue

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < conf_thresh:
            continue

        cx    = float(max_loc[0] + tw // 2)
        cy_ab = float(y_top + max_loc[1] + th // 2)

        # Sanity: must be above surface and not at very top
        if cy_ab + r >= surface_y - 5 or cy_ab - r <= 5:
            continue

        positions.append((fi, cx, cy_ab))
        confidences.append(max_val)

    positions.sort(key=lambda x: x[0])
    return positions, confidences


def template_d0_search(video_path, impact_frame, surface_y, px_per_mm,
                       lookback=40, r_min=50, r_max=110, r_step=5):
    """
    FIX 1 - Multi-scale template D0 estimation.

    Sweeps candidate radii r_min..r_max in r_step increments, runs
    template_track at each, and picks the radius whose matches score
    highest (mean_confidence x n_matches).  Only called when HoughCircles
    D0 is suspicious (< 60 px or < 3 detections), so it costs nothing for
    normal videos.

    Returns (D0_px, best_positions).
    """
    best_r, best_score, best_positions = None, -1.0, []
    for r in range(r_min, r_max + 1, r_step):
        positions, confs = template_track(
            video_path, impact_frame, surface_y,
            float(r), px_per_mm, lookback=lookback, conf_thresh=0.25)
        if len(confs) >= 2:
            score = float(np.mean(confs)) * len(confs)
            if score > best_score:
                best_score     = score
                best_r         = r
                best_positions = positions
    if best_r is None:
        return None, [], 0.0
    return float(best_r * 2), best_positions, best_score


def optical_flow_track(video_path, seed_frame, seed_cx, seed_cy,
                       surface_y, lookback=20):
    """
    IMPROVEMENT 1 (fallback) - Lucas-Kanade sparse optical flow.

    Starts from the last reliable HoughCircles detection and tracks the
    droplet centre backward (by running LK forward on reversed frame order).
    Used when template matching finds < 3 positions (very low contrast).

    Returns list of (frame_idx, cx, cy) in chronological order.
    """
    lk_params = dict(winSize=(31, 31), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               20, 0.03))
    frames_needed = list(range(seed_frame - lookback, seed_frame + 1))
    frames_needed = [fi for fi in frames_needed if fi >= 0]

    # Pre-load frames
    gray_stack = {}
    for fi in frames_needed:
        f = read_frame(video_path, fi)
        if f is not None:
            gray_stack[fi] = preprocess(f)

    if seed_frame not in gray_stack:
        return []

    # Backward tracking: seed -> earlier frames
    p0 = np.array([[seed_cx, seed_cy]], dtype=np.float32).reshape(-1, 1, 2)
    positions = [(seed_frame, float(seed_cx), float(seed_cy))]

    prev_fi   = seed_frame
    prev_gray = gray_stack[seed_frame]

    for fi in range(seed_frame - 1, seed_frame - lookback - 1, -1):
        if fi not in gray_stack:
            break
        curr_gray = gray_stack[fi]
        # LK from prev to curr (note: tracking backward in time)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **lk_params)
        if status is None or status[0][0] == 0:
            break
        cx, cy = float(p1[0, 0, 0]), float(p1[0, 0, 1])
        # Reject if drifted to top artifacts or below surface
        if cy <= 10 or cy + 5 >= surface_y:
            break
        positions.append((fi, cx, cy))
        p0        = p1
        prev_gray = curr_gray

    positions.sort(key=lambda x: x[0])
    return positions


# -- NEW: Impact-frame refinement ----------------------------------------------

def refine_impact_frame(video_path, impact_frame_est, surface_y,
                         px_per_mm, search_range=8):
    """
    IMPROVEMENT 2 - Find the true first-contact frame.

    Scans +/-search_range frames around the classical-CV estimate.  The first
    frame where the background-subtracted contact width exceeds 1 mm is
    declared the true impact frame.  Fixes +/-5-frame errors that inflate beta
    (by including early spreading frames) or deflate it (by missing them).

    Falls back to the estimate if no clear contact is found in the window.
    """
    min_contact_px = int(1.0 * px_per_mm)   # 1 mm threshold

    # Background: median of 8 frames before the estimate
    bg_frames = []
    for fi in range(impact_frame_est - 12, impact_frame_est - 2):
        f = read_frame(video_path, fi)
        if f is not None:
            bg_frames.append(f.astype(np.float32))
    if len(bg_frames) < 3:
        return impact_frame_est
    background = np.median(bg_frames, axis=0).astype(np.uint8)

    # Forward scan: find first frame with contact > 1 mm
    for fi in range(impact_frame_est - search_range,
                    impact_frame_est + search_range + 1):
        frame = read_frame(video_path, fi)
        if frame is None:
            continue
        w = contact_width_px(frame, background, surface_y)
        if w >= min_contact_px:
            return fi

    return impact_frame_est


# -- Phase processors ----------------------------------------------------------

def scan_pre_impact_d0(video_path, impact_frame, surface_y, px_per_mm,
                        lookback=40):
    """
    PASS 1 - HoughCircles backward scan (unchanged from v1).
    Reliable D0 measurement; position accuracy is secondary here.

    Returns:
      rows      - list of (fi, cx, cy, radius_px) tuples
      positions - list of (fi, cx, cy) tuples
      time_zero - frame index of first detection
    """
    R_MIN, R_MAX = 45, 110
    detections   = []

    for offset in range(2, lookback + 1):
        fi = impact_frame - offset
        if fi < 0:
            continue
        frame = read_frame(video_path, fi)
        if frame is None:
            continue
        gray = preprocess(frame)

        det = hough_detect(gray, min_r=R_MIN, max_r=R_MAX,
                           prefer_largest=True,
                           radius_min_accept=R_MIN, radius_max_accept=R_MAX)
        if det is None:
            continue
        hcx, hcy, hr = det
        if hcy + hr >= surface_y - 5 or hcy - hr <= 5:
            continue

        p = ellipse_params_from_circle(gray, hcx, hcy, hr)
        if p is None:
            continue
        detections.append((fi, p["cx"], p["cy"], p, hr))

    if not detections:
        return [], [], impact_frame - 20

    # Filter outlier cx values using the nearest-to-impact detections as anchor.
    # Take the median cx of the 5 detections closest to impact_frame (most reliable)
    # and reject anything more than 200px away — eliminates fixed nozzle/artifact
    # circles on the far side of the frame without disturbing valid large-droplet cx.
    detections_sorted_by_fi = sorted(detections, key=lambda d: abs(d[0] - impact_frame))
    anchor_cxs = [d[1] for d in detections_sorted_by_fi[:5]]
    anchor_cx = float(np.median(anchor_cxs))
    detections = [d for d in detections if abs(d[1] - anchor_cx) < 200]

    if not detections:
        return [], [], impact_frame - 20

    detections.sort(key=lambda x: x[0])
    time_zero = detections[0][0]
    rows, positions = [], []

    for fi, cx, cy, p, hr in detections:
        radius_px = p["D_px"] / 2.0 if p["D_px"] else float(hr)
        rows.append((fi, cx, cy, radius_px))
        positions.append((fi, cx, cy))

    return rows, positions, time_zero


def compute_u0(video_path, impact_frame, surface_y, D0_px, px_per_mm,
               pre_pos, pre_rows, lookback=40):
    """
    PASS 2 - Compute U0 via template matching, with optical flow fallback.

    Strategy:
      1. Run template matching with a disk sized to D0_px/2.
      2. If >= 3 confident positions, use median pairwise slope -> U0.
      3. If < 3 template positions, fallback to optical flow from the
         last HoughCircles detection.
      4. If still unreliable (< 3 pts, or speed > 2500 mm/s), return None.

    Also applies a D_px size filter to the v1 HoughCircles positions as a
    tertiary fallback (same as v1 but with tighter bounds).
    """
    radius_px = D0_px / 2.0

    # -- Attempt 1: template matching
    tmpl_pos, confs = template_track(
        video_path, impact_frame, surface_y,
        radius_px, px_per_mm, lookback=lookback)

    if len(tmpl_pos) >= 3:
        u0 = median_pairwise_speed(tmpl_pos, FPS_ACTUAL, px_per_mm)
        if u0 is not None and u0 >= 100:
            return u0, "template", tmpl_pos

    # -- Attempt 1b: template at reduced confidence (FIX 2)
    if len(tmpl_pos) < 3:
        tmpl_pos_lc, _ = template_track(
            video_path, impact_frame, surface_y,
            radius_px, px_per_mm, lookback=lookback, conf_thresh=0.20)
        if len(tmpl_pos_lc) >= 3:
            u0 = median_pairwise_speed(tmpl_pos_lc, FPS_ACTUAL, px_per_mm)
            if u0 is not None and u0 >= 100:
                return u0, "template_lc", tmpl_pos_lc

    # -- Attempt 2: optical flow from last HoughCircles seed
    if pre_pos:
        seed = pre_pos[-1]                   # chronologically last = closest to impact
        seed_fi, seed_cx, seed_cy = seed
        of_pos = optical_flow_track(
            video_path, seed_fi, seed_cx, seed_cy,
            surface_y, lookback=lookback)
        if len(of_pos) >= 3:
            u0 = median_pairwise_speed(of_pos, FPS_ACTUAL, px_per_mm)
            if u0 is not None and u0 >= 100:
                return u0, "optical_flow", of_pos

    # -- Attempt 3: filtered HoughCircles positions (v1 fallback)
    # pre_rows: list of (fi, cx, cy, radius_px) tuples
    if D0_px and pre_pos:
        vel_pos = []
        for row_tup, pos in zip(pre_rows, pre_pos):
            fi_r, cx_r, cy_r, r_px = row_tup
            D_px_r = r_px * 2 if r_px else None
            if (cy_r > 100 and D_px_r is not None
                    and 0.65 * D0_px <= D_px_r <= 1.50 * D0_px):
                vel_pos.append(pos)
        u0 = median_pairwise_speed(vel_pos, FPS_ACTUAL, px_per_mm)
        if u0 is not None:
            return u0, "hough_filtered", vel_pos

    return None, "none", []


def scan_rebound(video_path, liftoff_frame, surface_y, px_per_mm,
                 D0_px=None, search_frames=30):
    """
    Rebound phase - FIX 3: D0-constrained radius bounds.

    Constraining HoughCircles to +/-40% of the known droplet radius eliminates
    most false detections (nozzle shadow, surface reflections) that were
    causing U_rebound > U0 and suppressing COR computation.

    Returns:
      rows      - list of (fi, cx, cy, radius) tuples
      positions - list of (fi, cx, cy) tuples
    """
    if D0_px:
        r0    = D0_px / 2.0
        r_min = max(30, int(r0 * 0.60))
        r_max = min(130, int(r0 * 1.45))
    else:
        r_min, r_max = 40, 110

    rows, positions = [], []
    miss, down_count = 0, 0
    prev_cy = None

    for i in range(search_frames):
        fi = liftoff_frame + 1 + i
        frame = read_frame(video_path, fi)
        if frame is None:
            break
        gray = preprocess(frame)
        det  = hough_detect(gray, min_r=r_min, max_r=r_max,
                            prefer_largest=False,
                            radius_min_accept=r_min, radius_max_accept=r_max)
        if det is None:
            miss += 1
            if miss >= 3:
                break
            continue
        miss = 0
        hcx, hcy, hr = det
        if hcy + hr >= surface_y - 5:
            break
        p = ellipse_params_from_circle(gray, hcx, hcy, hr)
        if p is None:
            continue
        if prev_cy is not None:
            if p["cy"] >= prev_cy:
                down_count += 1
                if down_count >= 3:
                    break
            else:
                down_count = 0
        prev_cy = p["cy"]
        rows.append((fi, p["cx"], p["cy"], float(hr)))
        positions.append((fi, p["cx"], p["cy"]))
        if p["cy"] - p["major"] / 2 <= 5:
            break

    return rows, positions


def process_spreading(video_path, impact_frame, liftoff_frame,
                      surface_y, px_per_mm, time_zero):
    """
    Background-subtracted contact width - same as v1.
    Returns list of (fi, width_px, background) tuples.
    """
    bg_frames = []
    for fi in range(impact_frame - 5, impact_frame):
        f = read_frame(video_path, fi)
        if f is not None:
            bg_frames.append(f.astype(np.float32))
    if not bg_frames:
        return []
    background = np.median(bg_frames, axis=0).astype(np.uint8)
    rows = []
    cap  = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, impact_frame)
    for fi in range(impact_frame, liftoff_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        width_px = contact_width_px(frame, background, surface_y)
        rows.append((fi, width_px, background))
    cap.release()
    return rows


# -- Main ----------------------------------------------------------------------

def main(folder_filter=None, out_dir_override=None):
    features = json.loads(FEATURE_JSON.read_text())

    video_dir_map = {
        "02182026": (VIDEOS_02, PX_PER_MM),
        "03242026": (VIDEOS_03, PX_PER_MM),
    }
    if VIDEOS_05.exists():
        video_dir_map["05052026"] = (VIDEOS_05, PX_PER_MM_NEW)
    if VIDEOS_NEW1.exists():
        video_dir_map["05112026"] = (VIDEOS_NEW1, PX_PER_MM_NEW1)
    if VIDEOS_NEW2.exists():
        video_dir_map["05122026"] = (VIDEOS_NEW2, PX_PER_MM_NEW2)

    active_out_dir = Path(out_dir_override) if out_dir_override else OUT_DIR
    active_out_dir.mkdir(parents=True, exist_ok=True)
    active_summary_json = (
        active_out_dir / "summary_timeseries_v2.json"
        if out_dir_override
        else SUMMARY_JSON
    )

    # Create state_frames subdirectory
    state_frames_dir = active_out_dir / "state_frames"
    state_frames_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    print("Processing videos (v2: template matching + impact refinement)...\n")

    for f in features:
        folder_key = f["folder"]
        if folder_key not in video_dir_map:
            continue
        if folder_filter and folder_key != folder_filter:
            continue

        video_dir, px_per_mm = video_dir_map[folder_key]
        video_path    = str(video_dir / f["video"])
        surface_y     = f["surface_row_px"]
        impact_frame  = f["impact_frame"]
        liftoff_frame = f["liftoff_frame"]

        if liftoff_frame - impact_frame == 300:
            liftoff_frame = impact_frame + 60

        print(f"  {f['video']}", end="  ", flush=True)

        # -- 1a. Pass 1: HoughCircles -> D0
        pre_rows, pre_pos, time_zero = scan_pre_impact_d0(
            video_path, impact_frame, surface_y, px_per_mm)

        # pre_rows: list of (fi, cx, cy, radius_px) tuples
        D0_vals = [row_tup[3] * 2 for row_tup in pre_rows if row_tup[3] is not None]
        D0_px   = float(np.median(D0_vals[-5:])) if D0_vals else None
        D0_mm   = round(D0_px / px_per_mm, 4) if D0_px else None

        # -- FIX 1: template D0 cross-check
        d0_source   = "hough"
        _hough_too_large = D0_px is not None and D0_px > 155
        _tmpl_d0_needed = (D0_px is None
                           or D0_px < 60
                           or len(pre_rows) < 3
                           or (D0_px is not None and 82 < D0_px < 98)
                           or _hough_too_large)
        if _tmpl_d0_needed:
            tmpl_d0_px, _, _ = template_d0_search(
                video_path, impact_frame, surface_y, px_per_mm, lookback=40)
            if tmpl_d0_px is not None:
                if D0_px is None:
                    accept = True
                elif _hough_too_large:
                    accept = tmpl_d0_px < D0_px * 0.70
                else:
                    accept = tmpl_d0_px > D0_px * 1.30
                if accept:
                    D0_px     = tmpl_d0_px
                    D0_mm     = round(D0_px / px_per_mm, 4)
                    d0_source = "template"

        if not pre_rows:
            time_zero = max(0, impact_frame - 20)

        # -- 1b. Impact-frame refinement
        impact_frame_ref = refine_impact_frame(
            video_path, impact_frame, surface_y, px_per_mm, search_range=4)

        # -- 1c. Pass 2: template matching / optical flow -> U0
        if D0_px:
            U0, u0_method, vel_pos = compute_u0(
                video_path, impact_frame, surface_y, D0_px, px_per_mm,
                pre_pos, pre_rows, lookback=40)
        else:
            U0, u0_method, vel_pos = None, "none", []

        # -- 2. Spreading phase widths (for beta_max)
        spread_data = process_spreading(
            video_path, impact_frame_ref, liftoff_frame,
            surface_y, px_per_mm, time_zero)

        widths   = [sd[1] for sd in spread_data if sd[1] and sd[1] > 0]
        D_max_px = max(widths) if widths else None
        D_max_mm = round(D_max_px / px_per_mm, 4) if D_max_px else None
        beta_max = round(D_max_px / D0_px, 4) if (D_max_px and D0_px) else None

        # Build dict of spreading widths by frame index for fast lookup
        spread_width_by_frame = {sd[0]: sd[1] for sd in spread_data}
        # Background for spreading contact_width computation
        spread_background = spread_data[0][2] if spread_data else None

        # -- 3. Rebound detection (for COR)
        reb_raw, reb_pos = scan_rebound(
            video_path, liftoff_frame, surface_y, px_per_mm, D0_px=D0_px)

        U_rebound = median_pairwise_speed(reb_pos, FPS_ACTUAL, px_per_mm)
        COR = round(U_rebound / U0, 4) \
              if (U_rebound and U0 and U0 > 200 and U_rebound <= U0) else None

        # Build dicts for fast per-frame lookup
        reb_by_frame = {r[0]: r for r in reb_raw}           # {fi: (fi,cx,cy,hr)}
        pre_by_frame = {row_tup[0]: row_tup for row_tup in pre_rows}  # {fi: (fi,cx,cy,r)}

        # -- 4. Determine first_visible_frame
        if pre_pos:
            first_visible_frame = min(p[0] for p in pre_pos)
        else:
            first_visible_frame = max(0, impact_frame - 40)

        # -- 5. Determine total frame count
        cap_info = cv2.VideoCapture(video_path)
        total_video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_info.release()

        # Cap rebound at 200 frames after liftoff
        rebound_end_frame   = min(liftoff_frame + 200, total_video_frames - 1)
        last_frame_to_write = rebound_end_frame

        # -- 6. Prepare template for per-frame detection
        tmpl_radius  = D0_px / 2.0 if D0_px else 65.0
        disk_template = make_disk_template(tmpl_radius)
        th_t, tw_t   = disk_template.shape

        # If no spread background, recompute it
        if spread_background is None:
            bg_frames_sp = []
            for fi_bg in range(impact_frame_ref - 5, impact_frame_ref):
                frm = read_frame(video_path, fi_bg)
                if frm is not None:
                    bg_frames_sp.append(frm.astype(np.float32))
            spread_background = (
                np.median(bg_frames_sp, axis=0).astype(np.uint8)
                if bg_frames_sp else None
            )

        # -- 7. Main per-frame output loop
        csv_rows = []
        prev_phase    = None
        last_known_cx = None
        last_known_cy = None
        last_known_r  = None

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_visible_frame)
        current_frame_idx = first_visible_frame

        for fi in range(first_visible_frame, last_frame_to_write + 1):
            # Seek if VideoCapture fell out of sync
            if fi != current_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            current_frame_idx = fi + 1

            # Assign phase
            if fi < impact_frame_ref:
                phase = "falling"
            elif fi < liftoff_frame:
                phase = "spreading"
            else:
                phase = "rebounding"

            # State-change detection
            state_change = (phase != prev_phase)
            prev_phase   = phase

            # Save state-change frame as JPEG
            if state_change and ret and frame is not None:
                video_stem = Path(f["video"]).stem
                fname      = f"{video_stem}_frame{fi:05d}_{phase}.jpg"
                cv2.imwrite(str(state_frames_dir / fname), frame)

            time_ms = (fi - first_visible_frame) / FPS_ACTUAL * 1000

            if not ret or frame is None:
                csv_rows.append({
                    "frame": fi, "phase": phase,
                    "cx_px": None, "cy_px": None, "radius_px": None,
                    "spread_width_px": None,
                    "detection_method": "null", "confidence": 0.0,
                    "D_mm": None, "beta": None,
                    "velocity_mm_s": None, "time_ms": round(time_ms, 4),
                    "dist_travelled_px": None, "px_per_mm": px_per_mm,
                    "state_change": state_change,
                })
                continue

            gray = preprocess(frame)

            # -- Spreading phase
            if phase == "spreading":
                spread_w = spread_width_by_frame.get(fi, None)
                if spread_w is None and spread_background is not None:
                    spread_w = contact_width_px(frame, spread_background, surface_y)

                beta_val = None
                if spread_w and D0_px:
                    beta_val = round(spread_w / D0_px, 4)

                csv_rows.append({
                    "frame": fi, "phase": "spreading",
                    "cx_px": None, "cy_px": None, "radius_px": None,
                    "spread_width_px": spread_w,
                    "detection_method": "contact_width", "confidence": 1.0,
                    "D_mm": None, "beta": beta_val,
                    "velocity_mm_s": None, "time_ms": round(time_ms, 4),
                    "dist_travelled_px": None, "px_per_mm": px_per_mm,
                    "state_change": state_change,
                })
                # Reset last_known across spreading phase
                last_known_cx = None
                last_known_cy = None
                last_known_r  = None
                continue

            # -- Falling / rebounding: exhaustive detection chain
            det_cx, det_cy, det_r = None, None, None
            det_method = "null"
            det_conf   = 0.0

            # Step 1: Use pre-computed Hough detection if available
            if phase == "falling" and fi in pre_by_frame:
                row_tup = pre_by_frame[fi]
                det_cx, det_cy, det_r = row_tup[1], row_tup[2], row_tup[3]
                det_method = "hough"
                det_conf   = 1.0
            elif phase == "rebounding" and fi in reb_by_frame:
                rb = reb_by_frame[fi]
                det_cx, det_cy, det_r = rb[1], rb[2], rb[3]
                det_method = "hough"
                det_conf   = 1.0

            # Step 2: Template matching fallback
            if det_cx is None:
                y_top = max(0, int(tmpl_radius) - 5)
                y_bot = int(surface_y - tmpl_radius - 8)
                if y_bot - y_top >= th_t and gray.shape[1] >= tw_t:
                    search = gray[y_top: y_bot + th_t, :]
                    if search.shape[0] >= th_t and search.shape[1] >= tw_t:
                        result = cv2.matchTemplate(
                            search, disk_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        if max_val >= 0.30:
                            t_cx = float(max_loc[0] + tw_t // 2)
                            t_cy = float(y_top + max_loc[1] + th_t // 2)
                            if (t_cy + tmpl_radius < surface_y - 5
                                    and t_cy - tmpl_radius > 5):
                                det_cx     = t_cx
                                det_cy     = t_cy
                                det_r      = tmpl_radius
                                det_method = "template"
                                det_conf   = float(max_val)

            # Step 3: Last-known position fallback
            if det_cx is None and last_known_cx is not None:
                det_cx     = last_known_cx
                det_cy     = last_known_cy
                det_r      = last_known_r
                det_method = "last_known"
                det_conf   = 0.3

            # Update last_known
            if det_cx is not None:
                last_known_cx = det_cx
                last_known_cy = det_cy
                last_known_r  = det_r

            # Compute derived values
            D_mm_val = None
            beta_val = None
            if det_r is not None:
                D_mm_val = round(det_r * 2 / px_per_mm, 4)
                if D0_px:
                    beta_val = round(det_r * 2 / D0_px, 4)

            csv_rows.append({
                "frame": fi, "phase": phase,
                "cx_px":  round(det_cx, 1) if det_cx is not None else None,
                "cy_px":  round(det_cy, 1) if det_cy is not None else None,
                "radius_px": round(det_r, 1) if det_r is not None else None,
                "spread_width_px": None,
                "detection_method": det_method,
                "confidence": round(det_conf, 4),
                "D_mm": D_mm_val, "beta": beta_val,
                "velocity_mm_s": None,      # filled in next pass
                "time_ms": round(time_ms, 4),
                "dist_travelled_px": None,  # filled in next pass
                "px_per_mm": px_per_mm,
                "state_change": state_change,
            })

        cap.release()

        # -- 8. Fill velocity and dist_travelled_px
        prev_cx_v, prev_cy_v, prev_fi_v = None, None, None
        for row in csv_rows:
            if row["phase"] == "spreading":
                # Reset velocity continuity across spreading
                prev_cx_v, prev_cy_v, prev_fi_v = None, None, None
                continue
            cx_v = row["cx_px"]
            cy_v = row["cy_px"]
            fi_v = row["frame"]
            if (cx_v is not None and cy_v is not None
                    and prev_cx_v is not None and prev_cy_v is not None):
                fi_gap = fi_v - prev_fi_v
                if fi_gap > 0:
                    dx       = cx_v - prev_cx_v
                    dy       = cy_v - prev_cy_v
                    dist_px  = float(np.sqrt(dx ** 2 + dy ** 2))
                    vel_mm_s = dist_px * FPS_ACTUAL / (fi_gap * px_per_mm)
                    row["dist_travelled_px"] = round(dist_px, 3)
                    row["velocity_mm_s"]     = round(vel_mm_s, 2)
            if cx_v is not None and cy_v is not None:
                prev_cx_v = cx_v
                prev_cy_v = cy_v
                prev_fi_v = fi_v

        total_frames_written = len(csv_rows)

        # -- 9. Write per-video CSV
        out_csv = active_out_dir / f["video"].replace(".mp4", "_timeseries.csv")
        with open(str(out_csv), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)

        contact_ms   = round((liftoff_frame - impact_frame_ref) / FPS_ACTUAL * 1000, 4)
        impact_shift = impact_frame_ref - impact_frame

        pre_count    = sum(1 for r in csv_rows if r["phase"] == "falling")
        spread_count = sum(1 for r in csv_rows if r["phase"] == "spreading")
        reb_count    = sum(1 for r in csv_rows if r["phase"] == "rebounding")

        beta_flag = " *beta>5*" if (beta_max and beta_max > 5.0) else ""
        print(f"D0={D0_mm}mm({d0_source})  U0={U0}mm/s({u0_method})  "
              f"beta={beta_max}{beta_flag}  U_reb={U_rebound}mm/s  COR={COR}  "
              f"Dimp={impact_shift:+d}  pre/sp/reb="
              f"{pre_count}/{spread_count}/{reb_count}  "
              f"total={total_frames_written}")

        summary.append({
            "video":                f["video"],
            "folder":               f["folder"],
            "px_per_mm":            px_per_mm,
            "time_zero_frame":      time_zero,
            "impact_frame_orig":    impact_frame,
            "impact_frame_ref":     impact_frame_ref,
            "impact_frame_shift":   impact_shift,
            "liftoff_frame":        liftoff_frame,
            "contact_time_ms":      contact_ms,
            "D0_px":                round(D0_px, 3) if D0_px else None,
            "D0_mm":                D0_mm,
            "D_max_px":             D_max_px,
            "D_max_mm":             D_max_mm,
            "beta_max":             beta_max,
            "U0_mm_s":              U0,
            "U0_method":            u0_method,
            "U_rebound_mm_s":       U_rebound,
            "COR":                  COR,
            "pre_impact_frames":    pre_count,
            "spreading_frames":     spread_count,
            "rebound_frames":       reb_count,
            "D0_source":            d0_source,
            "beta_outlier":         beta_max is not None and beta_max > 5.0,
            "first_visible_frame":  first_visible_frame,
            "total_frames_written": total_frames_written,
        })

    active_summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nCSVs  -> {active_out_dir}/")
    print(f"Summary -> {active_summary_json}\n")

    # -- Final table
    print(f"  {'Video':<38} {'D0':>6} {'U0':>8} {'method':<14} "
          f"{'beta':>6} {'U_reb':>7} {'COR':>6} {'Dimp':>5}")
    print("  " + "-" * 100)
    for s in summary:
        print(f"  {s['video']:<38} "
              f"{str(s['D0_mm']):>6} "
              f"{str(s['U0_mm_s']):>8} "
              f"{s['U0_method']:<14} "
              f"{str(s['beta_max']):>6} "
              f"{str(s['U_rebound_mm_s']):>7} "
              f"{str(s['COR']):>6} "
              f"{s['impact_frame_shift']:>+5}")

    # -- Method breakdown
    from collections import Counter
    methods = Counter(s["U0_method"] for s in summary)
    print(f"\n  U0 method counts: {dict(methods)}")
    d0_srcs = Counter(s["D0_source"] for s in summary)
    print(f"  D0 source counts: {dict(d0_srcs)}")
    outliers = [s["video"] for s in summary if s.get("beta_outlier")]
    if outliers:
        print(f"  beta outliers (>5): {outliers}")

    cors = [s["COR"] for s in summary if s["COR"] and 0 < s["COR"] <= 1]
    if cors:
        print(f"  COR n={len(cors)}  mean={np.mean(cors):.3f}  "
              f"std={np.std(cors):.3f}  range=[{min(cors):.3f},{max(cors):.3f}]")

    # -- Validation vs supervisor
    manual = {
        "cainhsds2.mp4": {"D0": 2.348, "U0": None,   "beta": 1.8009},
        "caonly2.mp4":   {"D0": 1.928, "U0": 964.4,  "beta": 2.1580},
        "cainhtx1.mp4":  {"D0": 1.555, "U0": 1175.8, "beta": 2.0304},
    }
    sm = {s["video"]: s for s in summary}
    print("\n  -- Validation vs supervisor manual --")
    print(f"  {'Video':<22} {'Param':>6}  {'Manual':>8}  {'Auto':>8}  "
          f"{'Diff%':>8}  {'Method'}")
    print("  " + "-" * 72)
    for vid, mv in manual.items():
        s = sm.get(vid, {})
        pairs = [
            ("D0",   mv["D0"],   s.get("D0_mm"),    s.get("U0_method", "")),
            ("U0",   mv["U0"],   s.get("U0_mm_s"),  s.get("U0_method", "")),
            ("beta", mv["beta"], s.get("beta_max"), ""),
        ]
        for param, man_v, auto_v, meth in pairs:
            if man_v is None:
                continue
            if auto_v is None:
                print(f"  {vid:<22} {param:>6}  {man_v:>8.3f}  {'N/A':>8}  {'---':>8}")
            else:
                err  = (auto_v - man_v) / man_v * 100
                flag = "OK" if abs(err) < 10 else ("!" if abs(err) < 25 else "X")
                print(f"  {vid:<22} {param:>6}  {man_v:>8.3f}  {auto_v:>8.3f}  "
                      f"{err:>+8.1f}%  {flag}  {meth}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ellipse_timeseries_v2 -- droplet impact analysis")
    p.add_argument("--folder", default=None,
                   help="Only process this dataset folder key (e.g. 02182026, 05052026)")
    p.add_argument("--outdir", default=None,
                   help="Override output directory for CSVs and summary JSON")
    args = p.parse_args()
    main(folder_filter=args.folder, out_dir_override=args.outdir)
