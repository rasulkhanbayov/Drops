"""
ellipse_timeseries.py
=====================
Automates the supervisor's FIJI/ImageJ manual protocol:

  Pre-impact  : HoughCircles locates the droplet (robust, proven), then
                cv2.fitEllipse on the masked region gives Major, Minor, D,
                AR, Roundness, Circularity, Feret, Solidity (per frame).
                Backward scan from impact_frame finds all visible frames.
  Spreading   : background-subtracted contact-width → Length, Beta (per frame)
  Rebound     : same HoughCircles + fitEllipse, forward scan from liftoff,
                upward-motion check stops at re-impact or frame exit.

CLAHE contrast enhancement applied before every detection step.
Per-video CSV (same column layout as supervisor's ODS) +
summary_timeseries.json with D0, U0, beta_max, U_rebound, COR.

Output:
  /home/ubuntu/materials/timeseries/<video>_timeseries.csv
  /home/ubuntu/materials/summary_timeseries.json
"""

import cv2
import json
import csv
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
FPS_ACTUAL    = 2996.766489
PX_PER_MM     = 65.625
PX_PER_MM_NEW = 66.0            # 05052026 folder

VIDEOS_02 = Path("/home/ubuntu/materials/02182026")
VIDEOS_03 = Path("/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant")
VIDEOS_05 = Path("/home/ubuntu/materials/05052026")

FEATURE_JSON = Path("/home/ubuntu/materials/feature_table.json")
OUT_DIR      = Path("/home/ubuntu/materials/timeseries")
SUMMARY_JSON = Path("/home/ubuntu/materials/summary_timeseries.json")

OUT_DIR.mkdir(exist_ok=True)

CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

CSV_FIELDS = [
    "frame", "slice", "phase",
    "area", "mean", "min", "max",
    "X", "Y", "major", "minor", "angle",
    "circ", "feret", "feret_x", "feret_y", "feret_angle", "min_feret",
    "AR", "roundness", "solidity",
    "length",
    "D_px", "D_mm", "beta",
    "time_ms", "Y_dist_px",
    "dist_travelled_px", "velocity_px_s", "px_per_mm", "velocity_mm_s",
]


# ── Low-level helpers ────────────────────────────────────────────────────────

def read_frame(path, fi):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return CLAHE.apply(gray)


def hough_detect(gray, min_r=15, max_r=140, prefer_largest=False,
                 radius_min_accept=0, radius_max_accept=9999):
    """
    HoughCircles with cascading param2.
    prefer_largest=True  → return the largest radius circle (pre-impact droplet)
    prefer_largest=False → return the uppermost circle (rebound: droplet moving up)
    radius_min_accept / radius_max_accept: hard reject circles outside this range.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for p2 in [20, 15, 12, 10]:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=30, param1=50, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            # Filter by accepted radius range
            c = [ci for ci in c if radius_min_accept <= ci[2] <= radius_max_accept]
            if not c:
                continue
            if prefer_largest:
                best = sorted(c, key=lambda x: -x[2])[0]  # largest radius
            else:
                best = sorted(c, key=lambda x: x[1])[0]   # uppermost
            return float(best[0]), float(best[1]), float(best[2])
    return None


def ellipse_params_from_circle(gray, cx, cy, radius):
    """
    Create a binary mask from the HoughCircles result, find the contour,
    and run cv2.fitEllipse to get all ImageJ-equivalent shape descriptors.
    Falls back to circle-derived values if fitEllipse fails.
    """
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)

    # Threshold the actual image inside the circle region to get real contour
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick the largest contour inside the circle
    valid = [c for c in contours if cv2.contourArea(c) > 200]
    if valid and len(max(valid, key=cv2.contourArea)) >= 5:
        contour = max(valid, key=cv2.contourArea)
    else:
        # Fallback: use the circle mask itself as contour
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

    # Feret — max caliper between hull vertices (sampled to ≤40 pts)
    hull_pts = hull.reshape(-1, 2).astype(float)
    if len(hull_pts) > 40:
        idx = np.linspace(0, len(hull_pts) - 1, 40, dtype=int)
        hull_pts = hull_pts[idx]
    feret, fx, fy = 0.0, float(ex), float(ey)
    for i in range(len(hull_pts)):
        for j in range(i + 1, len(hull_pts)):
            d = float(np.linalg.norm(hull_pts[i] - hull_pts[j]))
            if d > feret:
                feret = d
                fx, fy = hull_pts[i]

    # Pixel intensity stats inside the Hough circle mask
    pixels   = gray[mask > 0]
    mean_val = float(pixels.mean()) if len(pixels) else 0.0
    min_val  = int(pixels.min())    if len(pixels) else 0
    max_val  = int(pixels.max())    if len(pixels) else 0

    D_px = float(np.sqrt(major * minor))   # supervisor's geometric mean

    return dict(
        cx=round(float(ex), 1), cy=round(float(ey), 1),
        area=round(area, 1), mean=round(mean_val, 3),
        min=min_val, max=max_val,
        major=round(float(major), 3), minor=round(float(minor), 3),
        angle=round(float(angle), 1),
        circ=round(circ, 3),
        feret=round(feret, 1), feret_x=round(fx, 1), feret_y=round(fy, 1),
        min_feret=round(float(minor), 3),
        AR=round(AR, 3), roundness=round(roundness, 3), solidity=round(solidity, 3),
        D_px=round(D_px, 3),
    )


def contact_width_px(frame, background, surface_y):
    diff  = cv2.absdiff(frame, background)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    band  = thresh[max(0, surface_y - 40): surface_y + 10, :]
    cols  = np.where(band.any(axis=0))[0]
    return int(cols[-1] - cols[0]) if len(cols) > 1 else 0


def fill_velocities(rows, positions, fps, px_per_mm):
    """Frame-to-frame Euclidean velocity — supervisor's sqrt formula."""
    for i in range(1, len(positions)):
        dx = positions[i][1] - positions[i - 1][1]
        dy = positions[i][2] - positions[i - 1][2]
        dist_px  = float(np.sqrt(dx ** 2 + dy ** 2))
        vel_mm_s = dist_px * fps / px_per_mm
        rows[i]["dist_travelled_px"] = round(dist_px, 3)
        rows[i]["velocity_px_s"]     = round(dist_px * fps, 2)
        rows[i]["velocity_mm_s"]     = round(vel_mm_s, 2)


def filter_falling_run(positions):
    """
    Keep only the longest consecutive run where cy is strictly increasing
    (droplet moving toward surface).  Removes nozzle/artifact frames that
    appear before the droplet enters the frame (their cy is stable/decreasing).
    Allows a gap of up to 2 frames between sequential frame indices.
    """
    if len(positions) < 2:
        return positions

    best_start, best_len = 0, 1
    curr_start, curr_len = 0, 1

    for i in range(1, len(positions)):
        fi_gap = positions[i][0] - positions[i - 1][0]
        cy_diff = positions[i][2] - positions[i - 1][2]
        if cy_diff > 0 and fi_gap <= 3:
            curr_len += 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
        else:
            curr_start = i
            curr_len = 1

    return positions[best_start: best_start + best_len]


def linreg_speed(positions, fps, px_per_mm, falling=False):
    """
    Speed in mm/s from Y vs frame_index.
    falling=True: apply monotonic-run filter then use median pairwise slope
    (Theil-Sen) — immune to a single noisy detection inflating the estimate.
    For rebound (falling=False), use median pairwise slope directly on all pts
    (robust against artifact frames with large cy jumps).
    """
    pts = filter_falling_run(positions) if falling else positions
    if len(pts) < 2:
        return None
    fis = np.array([p[0] for p in pts], dtype=float)
    ys  = np.array([p[2] for p in pts], dtype=float)
    if len(pts) >= 3:
        pair_slopes = [
            (ys[i + 1] - ys[i]) / (fis[i + 1] - fis[i])
            for i in range(len(pts) - 1) if fis[i + 1] > fis[i]
        ]
        if pair_slopes:
            slope = float(np.median(pair_slopes))
            return round(abs(slope) * fps / px_per_mm, 2)
    slope = np.polyfit(fis, ys, 1)[0]
    return round(abs(float(slope)) * fps / px_per_mm, 2)


def make_row(fi, slice_n, phase, p, surface_y, time_zero_ref, px_per_mm,
             length=0, beta=None, time_ref_frame=None):
    """Build one CSV row from ellipse params dict."""
    ref = time_ref_frame if time_ref_frame is not None else time_zero_ref
    time_ms = (fi - ref) / FPS_ACTUAL * 1000
    return {
        "frame": fi, "slice": slice_n, "phase": phase,
        "area": p["area"], "mean": p["mean"], "min": p["min"], "max": p["max"],
        "X": p["cx"], "Y": p["cy"],
        "major": p["major"], "minor": p["minor"], "angle": p["angle"],
        "circ": p["circ"], "feret": p["feret"],
        "feret_x": p["feret_x"], "feret_y": p["feret_y"],
        "feret_angle": p["angle"], "min_feret": p["min_feret"],
        "AR": p["AR"], "roundness": p["roundness"], "solidity": p["solidity"],
        "length": length,
        "D_px": p["D_px"], "D_mm": round(p["D_px"] / px_per_mm, 4),
        "beta": beta,
        "time_ms": round(time_ms, 4),
        "Y_dist_px": round(surface_y - p["cy"], 1),
        "dist_travelled_px": None, "velocity_px_s": None,
        "px_per_mm": px_per_mm, "velocity_mm_s": None,
    }


# ── Phase processors ─────────────────────────────────────────────────────────

def scan_pre_impact(video_path, impact_frame, surface_y, px_per_mm, lookback=40):
    """
    Backward scan from impact_frame to find all frames where droplet is
    visible and fully above surface. HoughCircles detects the circle;
    fitEllipse extracts shape parameters.
    Returns (rows, positions) in chronological order.
    """
    detections = []   # (fi, cx, cy, params_dict)

    # 4 µL droplet at 65.625 px/mm: diameter ≈ 1.97 mm → radius ≈ 65 px.
    # Accept radius 45–110 px to cover size variation; reject small artifacts.
    R_MIN, R_MAX = 45, 110

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

        # Must be clearly above surface and not cut off at top
        if hcy + hr >= surface_y - 5:
            continue
        if hcy - hr <= 5:
            continue

        p = ellipse_params_from_circle(gray, hcx, hcy, hr)
        if p is None:
            continue

        detections.append((fi, p["cx"], p["cy"], p))

    if not detections:
        return [], []

    # Sort chronologically and assign time_zero = first visible frame
    detections.sort(key=lambda x: x[0])
    time_zero = detections[0][0]

    rows      = []
    positions = []
    for slice_n, (fi, cx, cy, p) in enumerate(detections, start=1):
        row = make_row(fi, slice_n, "falling", p, surface_y,
                       time_zero_ref=time_zero, px_per_mm=px_per_mm)
        rows.append(row)
        positions.append((fi, cx, cy))

    fill_velocities(rows, positions, FPS_ACTUAL, px_per_mm)
    return rows, positions, time_zero


def scan_rebound(video_path, liftoff_frame, surface_y, px_per_mm,
                 search_frames=30):
    """
    Forward scan from liftoff, tracking upward-moving droplet.
    Stops at: no detection for 3 frames, droplet moving back toward surface
    for 3 consecutive frames, or droplet exits top of frame.
    """
    rows, positions = [], []
    miss       = 0
    down_count = 0
    prev_cy    = None
    slice_n    = 0

    for i in range(search_frames):
        fi = liftoff_frame + 1 + i
        frame = read_frame(video_path, fi)
        if frame is None:
            break
        gray = preprocess(frame)

        det = hough_detect(gray, min_r=40, max_r=110,
                           prefer_largest=False,
                           radius_min_accept=40, radius_max_accept=110)
        if det is None:
            miss += 1
            if miss >= 3:
                break
            continue
        miss = 0
        hcx, hcy, hr = det

        # Must still be above surface
        if hcy + hr >= surface_y - 5:
            break

        p = ellipse_params_from_circle(gray, hcx, hcy, hr)
        if p is None:
            continue

        # Require upward motion (cy decreasing toward top of frame)
        if prev_cy is not None:
            if p["cy"] >= prev_cy:
                down_count += 1
                if down_count >= 3:
                    break
            else:
                down_count = 0
        prev_cy = p["cy"]

        slice_n += 1
        row = make_row(fi, slice_n, "rebounding", p, surface_y,
                       time_zero_ref=liftoff_frame, px_per_mm=px_per_mm,
                       time_ref_frame=liftoff_frame)
        rows.append(row)
        positions.append((fi, p["cx"], p["cy"]))

        # Stop if exiting top of frame
        if p["cy"] - p["major"] / 2 <= 5:
            break

    fill_velocities(rows, positions, FPS_ACTUAL, px_per_mm)
    return rows, positions


def process_spreading(video_path, impact_frame, liftoff_frame,
                      surface_y, px_per_mm, time_zero):
    """Background-subtracted contact width for impact→liftoff frames."""
    bg_frames = []
    for fi in range(impact_frame - 5, impact_frame):
        f = read_frame(video_path, fi)
        if f is not None:
            bg_frames.append(f.astype(np.float32))
    if not bg_frames:
        return []
    background = np.median(bg_frames, axis=0).astype(np.uint8)

    rows = []
    cap  = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, impact_frame)

    for fi in range(impact_frame, liftoff_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        width_px = contact_width_px(frame, background, surface_y)
        time_ms  = (fi - time_zero) / FPS_ACTUAL * 1000
        rows.append({
            "frame": fi, "slice": fi - impact_frame + 1, "phase": "spreading",
            "area": None, "mean": None, "min": None, "max": None,
            "X": None, "Y": surface_y,
            "major": None, "minor": None, "angle": None,
            "circ": None, "feret": None,
            "feret_x": None, "feret_y": None, "feret_angle": None,
            "min_feret": None,
            "AR": None, "roundness": 0.0, "solidity": None,
            "length": width_px,
            "D_px": None, "D_mm": None, "beta": None,
            "time_ms": round(time_ms, 4),
            "Y_dist_px": 0,
            "dist_travelled_px": None, "velocity_px_s": None,
            "px_per_mm": px_per_mm, "velocity_mm_s": None,
        })
    cap.release()
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    features = json.loads(FEATURE_JSON.read_text())

    video_dir_map = {
        "02182026": (VIDEOS_02, PX_PER_MM),
        "03242026": (VIDEOS_03, PX_PER_MM),
    }
    if VIDEOS_05.exists():
        video_dir_map["05052026"] = (VIDEOS_05, PX_PER_MM_NEW)

    summary = []
    print("Processing videos...\n")

    for f in features:
        folder_key = f["folder"]
        if folder_key not in video_dir_map:
            continue

        video_dir, px_per_mm = video_dir_map[folder_key]
        video_path    = str(video_dir / f["video"])
        surface_y     = f["surface_row_px"]
        impact_frame  = f["impact_frame"]
        liftoff_frame = f["liftoff_frame"]

        # Clamp unreliable liftoff (300-frame cap → use 60-frame window)
        if liftoff_frame - impact_frame == 300:
            liftoff_frame = impact_frame + 60

        print(f"  {f['video']}", end="  ", flush=True)

        # ── 1. Pre-impact: backward scan ──────────────────────────────────
        result = scan_pre_impact(video_path, impact_frame, surface_y, px_per_mm)
        pre_rows, pre_pos, time_zero = result

        D0_vals  = [r["D_px"] for r in pre_rows if r["D_px"]]
        D0_px    = float(np.median(D0_vals[-5:])) if D0_vals else None
        D0_mm    = round(D0_px / px_per_mm, 4) if D0_px else None
        # Velocity: exclude top-of-frame artifacts (nozzle at cy < 100) and
        # detections with D_px far from D0 (artifact circles, spreading lamella).
        if D0_px:
            vel_pos = [
                pos for pos, row in zip(pre_pos, pre_rows)
                if pos[2] > 100
                and row["D_px"]
                and 0.65 * D0_px <= row["D_px"] <= 1.50 * D0_px
            ]
        else:
            vel_pos = [(fi, cx, cy) for fi, cx, cy in pre_pos if cy > 100]
        U0 = linreg_speed(vel_pos, FPS_ACTUAL, px_per_mm, falling=True)
        if U0 is not None and U0 > 2500:
            U0 = None  # implausible for these experiments (free-fall from ≤30 cm)

        # Fallback time_zero if no pre-impact frames found
        if not pre_rows:
            time_zero = max(0, impact_frame - 20)

        # ── 2. Spreading phase ────────────────────────────────────────────
        spread_rows = process_spreading(
            video_path, impact_frame, liftoff_frame, surface_y, px_per_mm, time_zero)

        widths   = [r["length"] for r in spread_rows if r["length"] and r["length"] > 0]
        D_max_px = max(widths) if widths else None
        D_max_mm = round(D_max_px / px_per_mm, 4) if D_max_px else None
        beta_max = round(D_max_px / D0_px, 4) if (D_max_px and D0_px) else None

        # ── 3. Rebound phase ──────────────────────────────────────────────
        reb_rows, reb_pos = scan_rebound(
            video_path, liftoff_frame, surface_y, px_per_mm)

        U_rebound = linreg_speed(reb_pos, FPS_ACTUAL, px_per_mm, falling=False)
        # COR must be in [0, 1]: U_rebound > U0 is non-physical (energy creation)
        COR = round(U_rebound / U0, 4) \
              if (U_rebound and U0 and U0 > 200 and U_rebound <= U0) else None

        # ── 4. Fill beta values now D0 is known ──────────────────────────
        for r in pre_rows:
            if r["D_px"] and D0_px:
                r["beta"] = round(r["D_px"] / D0_px, 4)
        for r in spread_rows:
            if r["length"] and D0_px:
                r["beta"] = round(r["length"] / D0_px, 4)
        for r in reb_rows:
            if r["D_px"] and D0_px:
                r["beta"] = round(r["D_px"] / D0_px, 4)

        # ── 5. Write per-video CSV ────────────────────────────────────────
        all_rows = pre_rows + spread_rows + reb_rows
        out_csv  = OUT_DIR / f["video"].replace(".mp4", "_timeseries.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

        contact_ms = round((liftoff_frame - impact_frame) / FPS_ACTUAL * 1000, 4)

        print(f"D0={D0_mm}mm  U0={U0}mm/s  β={beta_max}  "
              f"U_reb={U_rebound}mm/s  COR={COR}  "
              f"pre/sp/reb={len(pre_rows)}/{len(spread_rows)}/{len(reb_rows)}")

        summary.append({
            "video":             f["video"],
            "folder":            f["folder"],
            "px_per_mm":         px_per_mm,
            "time_zero_frame":   time_zero,
            "impact_frame":      impact_frame,
            "liftoff_frame":     liftoff_frame,
            "contact_time_ms":   contact_ms,
            "D0_px":             round(D0_px, 3) if D0_px else None,
            "D0_mm":             D0_mm,
            "D_max_px":          D_max_px,
            "D_max_mm":          D_max_mm,
            "beta_max":          beta_max,
            "U0_mm_s":           U0,
            "U_rebound_mm_s":    U_rebound,
            "COR":               COR,
            "pre_impact_frames": len(pre_rows),
            "spreading_frames":  len(spread_rows),
            "rebound_frames":    len(reb_rows),
        })

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nCSVs → {OUT_DIR}/")
    print(f"Summary → {SUMMARY_JSON}\n")

    # ── Final comparison table ────────────────────────────────────────────
    print(f"  {'Video':<35} {'D0(mm)':>8} {'U0(mm/s)':>10} {'β_max':>7} "
          f"{'U_reb':>10} {'COR':>7} {'pre/sp/reb':>12}")
    print("  " + "─" * 92)
    for s in summary:
        print(f"  {s['video']:<35} "
              f"{str(s['D0_mm']):>8} {str(s['U0_mm_s']):>10} "
              f"{str(s['beta_max']):>7} {str(s['U_rebound_mm_s']):>10} "
              f"{str(s['COR']):>7} "
              f"{str(s['pre_impact_frames'])+'/'+ str(s['spreading_frames'])+'/'+ str(s['rebound_frames']):>12}")

    cors = [s["COR"] for s in summary if s["COR"] and 0 < s["COR"] < 1.5]
    if cors:
        print(f"\n  COR (physical range 0–1.5): n={len(cors)}  "
              f"mean={np.mean(cors):.3f}  std={np.std(cors):.3f}  "
              f"range=[{min(cors):.3f},{max(cors):.3f}]")

    # ── Comparison against supervisor's manual values ─────────────────────
    manual = {
        "cainhsds2.mp4": {"D0": 2.348, "U0": None,  "beta": 1.8009},
        "caonly2.mp4":   {"D0": 1.928, "U0": 964.4, "beta": 2.1580},
        "cainhtx1.mp4":  {"D0": 1.555, "U0": 1175.8,"beta": 2.0304},
    }
    sm = {s["video"]: s for s in summary}
    print("\n  ── Validation vs supervisor manual (ODS) ──")
    print(f"  {'Video':<20} {'Param':>6}  {'Manual':>8}  {'Auto':>8}  {'Diff%':>8}")
    print("  " + "─" * 58)
    for vid, mv in manual.items():
        s = sm.get(vid, {})
        for param, mval in mv.items():
            if mval is None:
                continue
            aval = s.get({"D0":"D0_mm","U0":"U0_mm_s","beta":"beta_max"}[param])
            if aval:
                diff = (aval - mval) / mval * 100
                print(f"  {vid:<20} {param:>6}  {mval:>8.3f}  {aval:>8.4f}  {diff:>+8.1f}%")
            else:
                print(f"  {vid:<20} {param:>6}  {mval:>8.3f}  {'None':>8}")


if __name__ == "__main__":
    main()
