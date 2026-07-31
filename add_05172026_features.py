"""
Auto-detect surface_row and impact_frame for 05172026 videos,
then append entries to feature_table.json.
"""
import cv2
import json
import numpy as np
from pathlib import Path

FOLDER      = Path("/home/ubuntu/materials/05172026")
FPS_ACTUAL  = 2996.766489
PX_PER_MM   = 56.0   # same camera distance as 05122026 (farther back)
FEATURE_JSON = Path("/home/ubuntu/materials/feature_table.json")


def auto_surface_row(video_path: str, sample_frame: int = 0) -> int:
    """Detect surface row via horizontal Sobel edge on a background frame."""
    cap = cv2.VideoCapture(video_path)
    # Use median of first 10 frames as background
    frames = []
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if not frames:
        return 300
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    h = bg.shape[0]
    # Restrict search to 25%-85% of frame height (avoids nozzle and bottom edge)
    row_start = int(h * 0.25)
    row_end   = int(h * 0.85)
    roi = bg[row_start:row_end, :]
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobelx = np.abs(sobelx)
    row_energy = abs_sobelx.sum(axis=1)
    surface_row = int(row_start + int(np.argmax(row_energy)))
    return surface_row


def auto_impact_frame(video_path: str, surface_y: int) -> int:
    """Find impact frame: frame of max frame-diff in the lower surface band."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev, diffs = None, []
    band_top = max(0, surface_y - 20)
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            roi = gray[band_top:surface_y + 5, :]
            diff = float(np.abs(roi.astype(int) - prev[band_top:surface_y + 5, :].astype(int)).mean())
            diffs.append((i, diff))
        prev = gray
    cap.release()
    if not diffs:
        return 0
    return max(diffs, key=lambda x: x[1])[0]


def measure_pre_impact(video_path: str, impact: int, surface_y: int):
    """Sample up to 5 frames before impact, return median cx, cy, radius."""
    cap = cv2.VideoCapture(video_path)
    detections = []
    for offset in range(1, 6):
        fi = impact - offset * 3
        if fi < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=30, param1=50, param2=20,
                                   minRadius=20, maxRadius=120)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            for cx, cy, r in c:
                if cy + r < surface_y - 5 and cy - r > 5:
                    detections.append((float(cx), float(cy), float(r)))
                    break
    cap.release()
    if not detections:
        return None, None, None
    cxs = [d[0] for d in detections]
    cys = [d[1] for d in detections]
    rs  = [d[2] for d in detections]
    return float(np.median(cxs)), float(np.median(cys)), float(np.median(rs))


def find_liftoff_frame(video_path: str, impact: int, surface_y: int, window=300) -> int:
    cap = cv2.VideoCapture(video_path)
    liftoff = impact + window
    for i in range(impact + 10, impact + window):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=30, param1=50, param2=20,
                                   minRadius=15, maxRadius=120)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            for cx, cy, r in c:
                if cy + r < surface_y - 5:
                    liftoff = i
                    cap.release()
                    return liftoff
    cap.release()
    return liftoff


def measure_max_spread(video_path: str, impact: int, liftoff: int, surface_y: int):
    cap = cv2.VideoCapture(video_path)
    max_w = None
    for fi in range(impact, liftoff + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        band = thresh[max(0, surface_y - 10): surface_y + 5, :]
        cols = np.where(band.max(axis=0) > 0)[0]
        if len(cols) >= 5:
            w = float(cols[-1] - cols[0])
            if max_w is None or w > max_w:
                max_w = w
    cap.release()
    return max_w


def main():
    videos = sorted(FOLDER.glob("*.mp4"))
    if not videos:
        print("No mp4 files found in", FOLDER)
        return

    # Load existing feature_table
    existing = json.loads(FEATURE_JSON.read_text())
    # Remove any existing 05172026 entries
    existing = [e for e in existing if e.get("folder") != "05172026"]

    new_entries = []
    for vf in videos:
        cap_check = cv2.VideoCapture(str(vf))
        total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()
        if total_frames == 0:
            print(f"  [skip] {vf.name} — 0 frames (corrupted)")
            continue

        print(f"  {vf.name} ({total_frames} frames)...", flush=True)

        surface_y = auto_surface_row(str(vf))
        print(f"    surface_row={surface_y}", flush=True)

        impact = auto_impact_frame(str(vf), surface_y)
        print(f"    impact_frame={impact}", flush=True)

        liftoff = find_liftoff_frame(str(vf), impact, surface_y)
        contact_ms = round((liftoff - impact) / FPS_ACTUAL * 1000, 4)
        print(f"    liftoff_frame={liftoff}  contact_ms={contact_ms}", flush=True)

        cx, cy, r = measure_pre_impact(str(vf), impact, surface_y)
        d_mm = round(2 * r / PX_PER_MM, 4) if r is not None else None
        print(f"    D0={d_mm}mm  cx={cx}  cy={cy}  r={r}", flush=True)

        max_sw = measure_max_spread(str(vf), impact, liftoff, surface_y)
        max_sw_mm = round(max_sw / PX_PER_MM, 4) if max_sw is not None else None
        beta_max = round(max_sw / (2 * r), 4) if (max_sw and r) else None
        print(f"    beta_max={beta_max}  max_spread_px={max_sw}", flush=True)

        entry = {
            "folder": "05172026",
            "video": vf.name,
            "surface_row_px": surface_y,
            "impact_frame": impact,
            "liftoff_frame": liftoff,
            "contact_time_ms": contact_ms,
            "pre_impact_cx_px": cx,
            "pre_impact_cy_px": cy,
            "pre_impact_radius_px": r,
            "pre_impact_diameter_mm": d_mm,
            "max_spread_width_px": max_sw,
            "max_spread_width_mm": max_sw_mm,
            "max_spread_factor": beta_max,
            "impact_velocity_px_per_frame": None,
            "impact_velocity_mm_per_s": None,
        }
        new_entries.append(entry)
        print()

    all_entries = existing + new_entries
    FEATURE_JSON.write_text(json.dumps(all_entries, indent=2))
    print(f"Done. Added {len(new_entries)} entries for 05172026.")
    print(f"Total entries in feature_table.json: {len(all_entries)}")


if __name__ == "__main__":
    main()
