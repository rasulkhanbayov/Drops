"""
Droplet segmentation using SAM2 video predictor.
- Uses OpenCV background subtraction to auto-detect the initial droplet position
- Feeds that as a point prompt to SAM2 video predictor
- Propagates the mask forward through the video
- Uses connected components on the propagated mask to detect splits after impact
- Outputs centroid + percentage to CSV
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


# ── helpers ────────────────────────────────────────────────────────────────

def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def contour_circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def connected_components(mask: np.ndarray, min_area: int):
    """
    Split a binary mask into connected components.
    Returns list of (component_mask, area) sorted by area descending.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    components = []
    for label in range(1, n):  # skip background label 0
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((labels == label, area))
    components.sort(key=lambda x: x[1], reverse=True)
    return components


def find_reference_droplet(video_path: str, bg_frames: int,
                            diff_thresh: int, min_area: int,
                            circ_thresh: float, margin: int):
    """
    Scans the video to find the first frame where the droplet is fully visible.
    Returns (frame_idx, cx, cy, area_px) and the static background image.
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build background from first bg_frames
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

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def extract_frames_to_dir(video_path: str, out_dir: str) -> int:
    """Extract all video frames as JPEG files. SAM2 video predictor requires this."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    return total


# ── main ───────────────────────────────────────────────────────────────────

def main(video_path: str, output_csv: str,
         checkpoint: str, model_cfg: str,
         bg_frames: int, diff_thresh: int,
         min_area: int, circ_thresh: float, margin: int):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            "Download from https://github.com/facebookresearch/sam2#model-checkpoints"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cpu":
        print("WARNING: SAM2 on CPU is extremely slow. GPU strongly recommended.")

    cap_info = cv2.VideoCapture(video_path)
    frame_w     = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()
    print(f"Video  : {video_path}")
    print(f"Size   : {frame_w}x{frame_h}  |  {total_frames} frames  |  {fps:.1f} fps")

    # ── Step 1: locate reference frame ────────────────────────────────────
    print(f"\nScanning for first fully-visible droplet frame...")
    ref_idx, ref_cx, ref_cy, ref_area_cv = find_reference_droplet(
        video_path, bg_frames, diff_thresh, min_area, circ_thresh, margin
    )
    if ref_idx is None:
        print("ERROR: Could not detect a fully-visible droplet. "
              "Try lowering --min_area or --diff_thresh.")
        return
    print(f"  Reference frame : {ref_idx}")
    print(f"  Droplet centroid: ({ref_cx}, {ref_cy})")
    print(f"  OpenCV area est : {ref_area_cv} px²")

    # ── Step 2: extract frames ────────────────────────────────────────────
    frame_dir = tempfile.mkdtemp(prefix="sam2_droplet_")
    print(f"\nExtracting {total_frames} frames to {frame_dir} ...")
    extract_frames_to_dir(video_path, frame_dir)
    print("Extraction complete.")

    # ── Step 3: SAM2 tracking ─────────────────────────────────────────────
    rows = []
    reference_area = None  # set from SAM2 mask on ref frame

    print(f"\nLoading SAM2 model from {checkpoint} ...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    split_min_area = max(50, min_area // 3)  # smaller threshold for split fragments

    try:
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            state = predictor.init_state(video_path=frame_dir)

            # Prompt SAM2 with the centroid found via OpenCV
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=ref_idx,
                obj_id=1,
                points=np.array([[ref_cx, ref_cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

            print("Propagating masks through video...")
            for frame_idx, obj_ids, masks_logits in predictor.propagate_in_video(state):
                # Skip frames before the reference (no droplet yet)
                if frame_idx < ref_idx:
                    continue

                # masks_logits: (num_objects, 1, H, W)
                mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                if not mask.any():
                    continue

                # Set reference area from first SAM2 mask (ref frame)
                if reference_area is None:
                    reference_area = int(mask.sum())
                    print(f"  SAM2 reference area: {reference_area} px²")

                # Split mask into connected components (handles post-impact fragments)
                components = connected_components(mask, min_area=split_min_area)
                if not components:
                    continue

                for drop_id, (comp_mask, area) in enumerate(components, start=1):
                    centroid = mask_centroid(comp_mask)
                    if centroid is None:
                        continue
                    cx, cy = centroid
                    pct = round(area / reference_area * 100, 2)
                    rows.append({
                        'frame':      frame_idx,
                        'drop_id':    drop_id,
                        'cx':         cx,
                        'cy':         cy,
                        'area_px':    area,
                        'percentage': pct,
                    })

                if frame_idx % 100 == 0:
                    n_drops = len(components)
                    print(f"  Frame {frame_idx}/{total_frames}  |  {n_drops} drop(s) detected")

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
        print("Temporary frame directory removed.")

    # ── Step 4: write CSV ─────────────────────────────────────────────────
    fieldnames = ['frame', 'drop_id', 'cx', 'cy', 'area_px', 'percentage']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.")
    print(f"  Reference frame : {ref_idx}")
    print(f"  Reference area  : {reference_area} px²")
    print(f"  Total rows      : {len(rows)}")
    print(f"  Output CSV      : {output_csv}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Droplet segmentation using SAM2 video predictor'
    )
    parser.add_argument('--video',       default='cainhsds1.mp4',
                        help='Input video file path')
    parser.add_argument('--output',      default='droplet_results_sam2.csv',
                        help='Output CSV file path')
    parser.add_argument('--checkpoint',  default='checkpoints/sam2.1_hiera_large.pt',
                        help='Path to SAM2 model checkpoint (.pt file)')
    parser.add_argument('--model_cfg',   default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='SAM2 model config YAML (relative to sam2 package root)')
    parser.add_argument('--bg_frames',   type=int,   default=30,
                        help='Frames used to build static background (default: 30)')
    parser.add_argument('--diff_thresh', type=int,   default=25,
                        help='Pixel diff threshold for foreground detection (default: 25)')
    parser.add_argument('--min_area',    type=int,   default=150,
                        help='Minimum droplet area in pixels (default: 150)')
    parser.add_argument('--circ_thresh', type=float, default=0.3,
                        help='Minimum circularity 0-1 for initial detection (default: 0.3)')
    parser.add_argument('--margin',      type=int,   default=3,
                        help='Edge margin px for fully-in-frame check (default: 3)')
    args = parser.parse_args()

    main(args.video, args.output, args.checkpoint, args.model_cfg,
         args.bg_frames, args.diff_thresh, args.min_area,
         args.circ_thresh, args.margin)
