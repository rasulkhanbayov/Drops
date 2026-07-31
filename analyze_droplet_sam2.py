"""
Droplet segmentation using SAM2 video predictor.

Full per-frame output from first detection onwards:
  - One CSV row per frame continuously — no gaps.
  - If SAM2 loses the mask, falls back to: HoughCircles → template matching
    → last-known position. Only writes null if all fallbacks exhausted.

Outputs: frame, drop_id, cx, cy, area_px, percentage
"""

import os
import csv
import shutil
import tempfile
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    raise ImportError(
        "SAM2 is not installed. See README_sam2.md for setup instructions.\n"
        "Quick install: pip install git+https://github.com/facebookresearch/sam2.git"
    )


CLAHE       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
FPS_ACTUAL  = 2996.766489  # true capture rate — OpenCV reports ~60, ignore it


# ── helpers ────────────────────────────────────────────────────────────────────

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return CLAHE.apply(gray)


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    return int(xs.mean()), int(ys.mean())


def contour_circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def connected_components(mask: np.ndarray, min_area: int):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    components = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((labels == label, area))
    components.sort(key=lambda x: x[1], reverse=True)
    return components


def hough_detect_droplet(gray, min_r=15, max_r=140, surface_y=None):
    """HoughCircles with progressive param2 relaxation. Filters below surface."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for p2 in [20, 15, 12, 10]:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=30, param1=50, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is None:
            continue
        c = np.round(circles[0]).astype(int)
        for cx, cy, r in sorted(c, key=lambda x: x[1]):  # uppermost first
            if surface_y is not None and cy + r >= surface_y - 5:
                continue
            if cy - r <= 5:
                continue
            return float(cx), float(cy), float(r)
    return None, None, None


def make_disk_template(radius_px):
    """Dark droplet blob + bright caustic ring template for shadowgraphy."""
    r = int(round(radius_px))
    size = 2 * r + 1
    tmpl = np.full((size, size), 180, dtype=np.uint8)
    cv2.circle(tmpl, (r, r), r, 40, -1)
    cv2.circle(tmpl, (r, r), r, 255, max(1, r // 6))
    return tmpl


def template_detect(gray, tmpl_radius, surface_y):
    """Template matching for droplet. Returns (cx, cy, conf) or (None, None, 0)."""
    disk = make_disk_template(tmpl_radius)
    th, tw = disk.shape
    y_top = max(0, int(tmpl_radius) - 5)
    y_bot = int(surface_y - tmpl_radius - 8)
    if y_bot - y_top < th or gray.shape[1] < tw:
        return None, None, 0.0
    search = gray[y_top: y_bot + th, :]
    if search.shape[0] < th or search.shape[1] < tw:
        return None, None, 0.0
    result = cv2.matchTemplate(search, disk, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.20:
        return None, None, 0.0
    t_cx = float(max_loc[0] + tw // 2)
    t_cy = float(y_top + max_loc[1] + th // 2)
    if t_cy + tmpl_radius >= surface_y - 5 or t_cy - tmpl_radius <= 5:
        return None, None, 0.0
    return t_cx, t_cy, float(max_val)


def find_reference_droplet(video_path, bg_frames, diff_thresh, min_area,
                            circ_thresh, margin):
    """Scan video for first fully-visible droplet. Returns (frame_idx, cx, cy, area)."""
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_list = []
    for _ in range(bg_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bg_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))

    if not bg_list:
        cap.release()
        return None, None, None, None

    background = np.mean(bg_list, axis=0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frame_idx = len(bg_list)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background)
        _, thresh = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            if contour_circularity(c) < circ_thresh:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if (x > margin and y > margin
                    and (x + w) < (frame_w - margin)
                    and (y + h) < (frame_h - margin)):
                M = cv2.moments(c)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cap.release()
                return frame_idx, cx, cy, int(area)
        frame_idx += 1

    cap.release()
    return None, None, None, None


def extract_frames_to_dir(video_path, out_dir, frame_step=1):
    """Extract every frame_step-th frame as JPEG. Returns total original frames."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{written:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    cap.release()
    return total


def read_frame_from_video(video_path, frame_idx):
    """Read a single frame by index from original video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── main ───────────────────────────────────────────────────────────────────────

def main(video_path, output_csv, checkpoint, model_cfg,
         bg_frames, diff_thresh, min_area, circ_thresh, margin,
         frame_step=1, surface_y=None, px_per_mm=65.625,
         impact_frame=None, liftoff_frame=None):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cpu":
        print("WARNING: SAM2 on CPU is extremely slow.")

    cap_info = cv2.VideoCapture(video_path)
    frame_w      = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()
    print(f"Video  : {video_path}")
    print(f"Size   : {frame_w}x{frame_h}  |  {total_frames} frames  |  {fps:.1f} fps")
    if frame_step > 1:
        print(f"Frame step : {frame_step} (~{total_frames // frame_step} sampled frames)")

    # -- Step 1: locate reference frame ----------------------------------------
    print("\nScanning for first fully-visible droplet frame...")
    ref_idx, ref_cx, ref_cy, ref_area_cv = find_reference_droplet(
        video_path, bg_frames, diff_thresh, min_area, circ_thresh, margin)
    if ref_idx is None:
        print("ERROR: Could not detect droplet. Try lowering --min_area or --diff_thresh.")
        return
    print(f"  Reference frame : {ref_idx}  centroid: ({ref_cx}, {ref_cy})  area: {ref_area_cv} px²")

    # Auto-estimate surface_y if not provided: bottom 20% of frame
    if surface_y is None:
        surface_y = int(frame_h * 0.80)
        print(f"  surface_y not provided — using {surface_y} (80% of frame height)")

    # Estimate droplet radius from reference area for template/hough fallbacks
    tmpl_radius = max(15.0, float(np.sqrt(ref_area_cv / np.pi)))
    hough_min_r = max(10, int(tmpl_radius * 0.5))
    hough_max_r = min(140, int(tmpl_radius * 1.8))
    print(f"  Estimated radius: {tmpl_radius:.1f} px  "
          f"(hough range {hough_min_r}–{hough_max_r})")

    # -- Step 2: extract frames for SAM2 ---------------------------------------
    frame_dir = tempfile.mkdtemp(prefix="sam2_droplet_")
    print(f"\nExtracting frames to {frame_dir} ...")
    extract_frames_to_dir(video_path, frame_dir, frame_step)
    print("Extraction complete.")

    ref_idx_sam = ref_idx // frame_step  # reference in subsampled sequence

    # -- Step 3: SAM2 tracking -------------------------------------------------
    # sam2_results: {original_frame_idx -> list of (drop_id, cx, cy, area_px)}
    sam2_results = {}
    reference_area = None

    print(f"\nLoading SAM2 model from {checkpoint} ...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    split_min_area = max(50, min_area // 3)

    try:
        with torch.inference_mode(), torch.autocast(device_type=device,
                                                     dtype=torch.bfloat16):
            state = predictor.init_state(video_path=frame_dir)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=ref_idx_sam,
                obj_id=1,
                points=np.array([[ref_cx, ref_cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

            print("Propagating masks through video...")
            for sam_idx, obj_ids, masks_logits in predictor.propagate_in_video(state):
                if sam_idx < ref_idx_sam:
                    continue

                mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                actual_frame = sam_idx * frame_step

                if not mask.any():
                    # SAM2 lost mask — record as empty so fallback triggers later
                    sam2_results[actual_frame] = []
                    continue

                if reference_area is None:
                    reference_area = int(mask.sum())
                    print(f"  SAM2 reference area: {reference_area} px²")

                components = connected_components(mask, min_area=split_min_area)
                if not components:
                    sam2_results[actual_frame] = []
                    continue

                detections = []
                for drop_id, (comp_mask, area) in enumerate(components, start=1):
                    cx, cy = mask_centroid(comp_mask)
                    if cx is None:
                        continue
                    detections.append((drop_id, cx, cy, area))
                sam2_results[actual_frame] = detections

                if sam_idx % 100 == 0:
                    print(f"  Frame {actual_frame}/{total_frames}  |  "
                          f"{len(detections)} drop(s) detected")
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
        print("Temporary frame directory removed.")

    if reference_area is None:
        reference_area = ref_area_cv  # fallback to OpenCV estimate

    # -- Step 4: per-frame output with exhaustive fallback chain ---------------
    # For frame_step > 1, SAM2 only has results at every frame_step-th frame.
    # For frames between SAM2 samples, we always run the fallback chain.

    fieldnames = ['frame', 'drop_id', 'cx', 'cy', 'area_px', 'percentage',
                  'detection_method']
    rows = []

    last_known_cx  = None
    last_known_cy  = None
    last_known_r   = None

    print(f"\nBuilding per-frame output from frame {ref_idx} to {total_frames - 1}...")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    current_fi = ref_idx

    for fi in range(ref_idx, total_frames):
        # Read frame from video sequentially
        if fi != current_fi:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        current_fi = fi + 1

        # Check if SAM2 has a result for this frame (only at sampled frames)
        is_sampled = (fi % frame_step == 0)
        sam2_dets  = sam2_results.get(fi, None) if is_sampled else None

        if sam2_dets is not None and len(sam2_dets) > 0:
            # SAM2 succeeded — use its detections
            for drop_id, cx, cy, area in sam2_dets:
                pct = round(area / reference_area * 100, 2)
                rows.append({
                    'frame': fi, 'drop_id': drop_id,
                    'cx': cx, 'cy': cy,
                    'area_px': area, 'percentage': pct,
                    'detection_method': 'sam2',
                })
            # Update last_known from largest component (drop_id=1)
            last_known_cx = float(sam2_dets[0][1])
            last_known_cy = float(sam2_dets[0][2])
            last_known_r  = float(np.sqrt(sam2_dets[0][3] / np.pi))
            continue

        # SAM2 failed or this is a between-sample frame — run fallback chain
        if not ret or frame is None:
            rows.append({'frame': fi, 'drop_id': 0,
                         'cx': None, 'cy': None,
                         'area_px': None, 'percentage': None,
                         'detection_method': 'null'})
            continue

        gray = preprocess(frame)
        det_cx, det_cy, det_r = None, None, None
        method = 'null'

        # Determine phase for this frame (needs impact/liftoff passed in)
        in_spreading = (impact_frame is not None and liftoff_frame is not None
                        and impact_frame <= fi <= liftoff_frame)

        # Max allowed displacement from last-known position per frame-gap.
        # A real droplet at ~1 m/s moves at most ~2 px/frame at 3000 fps.
        # Allow 6x margin (12 px/frame) to cover fast-moving or frame_step>1 cases.
        # Detections further than this from last-known are frame artifacts.
        max_disp_per_frame = 12.0 * frame_step

        # Fallback 1: HoughCircles — skip during spreading phase.
        # During spreading the droplet is a flat lamella: HoughCircles latches onto
        # fixed frame artifacts (nozzle, edge shadows) at a completely different
        # coordinate than the SAM2 centroid, causing large spurious velocity spikes.
        if not in_spreading:
            h_cx, h_cy, h_r = hough_detect_droplet(
                gray, min_r=hough_min_r, max_r=hough_max_r, surface_y=surface_y)
            if h_cx is not None:
                # Spatial consistency check: reject if too far from last-known position
                if last_known_cx is not None:
                    disp = np.sqrt((h_cx - last_known_cx)**2 + (h_cy - last_known_cy)**2)
                    if disp > max_disp_per_frame:
                        h_cx = None  # reject — likely a fixed artifact, not the droplet
                if h_cx is not None:
                    det_cx, det_cy, det_r = h_cx, h_cy, h_r
                    method = 'hough'

        # Fallback 2: Template matching — also skip during spreading (same reason),
        # and apply same spatial consistency check.
        if det_cx is None and not in_spreading:
            t_cx, t_cy, t_conf = template_detect(gray, tmpl_radius, surface_y)
            if t_cx is not None:
                if last_known_cx is not None:
                    disp = np.sqrt((t_cx - last_known_cx)**2 + (t_cy - last_known_cy)**2)
                    if disp > max_disp_per_frame:
                        t_cx = None
                if t_cx is not None:
                    det_cx, det_cy, det_r = t_cx, t_cy, tmpl_radius
                    method = 'template'

        # Fallback 3: Last-known position (always safe — same coordinate system)
        if det_cx is None and last_known_cx is not None:
            det_cx, det_cy, det_r = last_known_cx, last_known_cy, last_known_r
            method = 'last_known'

        if det_cx is not None:
            area = int(np.pi * det_r ** 2)
            pct  = round(area / reference_area * 100, 2)
            rows.append({
                'frame': fi, 'drop_id': 1,
                'cx': int(round(det_cx)), 'cy': int(round(det_cy)),
                'area_px': area, 'percentage': pct,
                'detection_method': method,
            })
            last_known_cx = det_cx
            last_known_cy = det_cy
            last_known_r  = det_r
        else:
            rows.append({'frame': fi, 'drop_id': 0,
                         'cx': None, 'cy': None,
                         'area_px': None, 'percentage': None,
                         'detection_method': 'null'})

    cap.release()

    # -- Step 5: compute per-frame velocity per drop_id ------------------------
    # Velocity is only valid when consecutive detections use the same method.
    # Cross-method transitions (e.g. sam2->hough) can have coordinate system
    # discontinuities that produce physically meaningless velocity spikes.
    last_pos = {}     # drop_id -> (frame, cx, cy, method)
    for row in rows:
        did    = row['drop_id']
        cx     = row['cx']
        cy     = row['cy']
        fi     = row['frame']
        method = row['detection_method']
        row['distance_px']       = None
        row['velocity_px_per_s'] = None
        row['velocity_mm_s']     = None
        if cx is not None and cy is not None and did != 0:
            if did in last_pos:
                prev_fi, prev_cx, prev_cy, prev_method = last_pos[did]
                frame_gap = fi - prev_fi
                # Null velocity on method change — coordinates may be inconsistent
                if frame_gap > 0 and method == prev_method:
                    dist = float(np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2))
                    time_s = frame_gap / FPS_ACTUAL
                    vel_px = dist / time_s
                    row['distance_px']       = round(dist, 3)
                    row['velocity_px_per_s'] = round(vel_px, 3)
                    row['velocity_mm_s']     = round(vel_px / px_per_mm, 3)
            last_pos[did] = (fi, cx, cy, method)

    # -- Step 6: assign phase per frame ----------------------------------------
    for row in rows:
        fi = row['frame']
        if impact_frame is not None and liftoff_frame is not None:
            if fi < impact_frame:
                row['phase'] = 'falling'
            elif fi <= liftoff_frame:
                row['phase'] = 'spreading'
            else:
                row['phase'] = 'rebounding'
        else:
            row['phase'] = None

    # -- Step 7: write CSV -----------------------------------------------------
    fieldnames = ['frame', 'phase', 'drop_id', 'cx', 'cy', 'area_px', 'percentage',
                  'detection_method', 'distance_px', 'velocity_px_per_s',
                  'velocity_mm_s']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    method_counts = {}
    for r in rows:
        m = r['detection_method']
        method_counts[m] = method_counts.get(m, 0) + 1

    print(f"\nDone.")
    print(f"  Reference frame  : {ref_idx}")
    print(f"  Reference area   : {reference_area} px²")
    print(f"  Total rows       : {len(rows)}  (expected {total_frames - ref_idx})")
    print(f"  Detection method counts: {method_counts}")
    print(f"  Output CSV       : {output_csv}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Droplet segmentation: SAM2 + HoughCircles + template fallback'
    )
    parser.add_argument('--video',       default='cainhsds1.mp4')
    parser.add_argument('--output',      default='droplet_results_sam2.csv')
    parser.add_argument('--checkpoint',  default='/data/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_cfg',   default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--bg_frames',   type=int,   default=30)
    parser.add_argument('--diff_thresh', type=int,   default=25)
    parser.add_argument('--min_area',    type=int,   default=150)
    parser.add_argument('--circ_thresh', type=float, default=0.3)
    parser.add_argument('--margin',      type=int,   default=3)
    parser.add_argument('--frame_step',  type=int,   default=1,
                        help='Set to 4 for videos >= 5000 frames')
    parser.add_argument('--surface_y',   type=int,   default=None,
                        help='Surface row in pixels (auto-estimated if not provided)')
    parser.add_argument('--px_per_mm',     type=float, default=65.625,
                        help='Pixel-per-mm calibration (65.625 for 02182026/03242026, '
                             '66.0 for 05052026, 66.5 for 05112026, 56.0 for 05122026/05172026)')
    parser.add_argument('--impact_frame',  type=int,   default=None,
                        help='Impact frame index for phase labelling')
    parser.add_argument('--liftoff_frame', type=int,   default=None,
                        help='Liftoff frame index for phase labelling')
    args = parser.parse_args()

    main(args.video, args.output, args.checkpoint, args.model_cfg,
         args.bg_frames, args.diff_thresh, args.min_area,
         args.circ_thresh, args.margin, args.frame_step, args.surface_y,
         args.px_per_mm, args.impact_frame, args.liftoff_frame)
