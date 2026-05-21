"""
Droplet segmentation using OpenCV background subtraction.
Detects droplets per frame, tracks splits, outputs centroid + percentage to CSV.

v2 improvements over v1:
  - Persistent drop ID tracking across frames (greedy nearest-neighbour centroid matching)
  - Two-phase detection: strict filters for finding reference frame, relaxed post-reference
    to catch deformed spreading shapes during impact
  - in_frame column flags fragments that exit the frame boundary
  - --folder batch mode processes every .mp4 in a directory
"""

import os
import csv
import argparse
import cv2
import numpy as np
from pathlib import Path


# ── drop ID tracker ─────────────────────────────────────────────────────────

class DropletTracker:
    """
    Assigns consistent integer IDs to drops across frames.
    Uses greedy nearest-neighbour matching on centroids.
    IDs are never reused once a drop disappears.
    Drop 1 is always the largest drop in the reference frame;
    new drops produced by splitting receive the next available ID.
    """

    def __init__(self, max_dist: float = 100.0):
        self.max_dist = max_dist
        self.next_id  = 1
        self.active   = {}   # drop_id -> (cx, cy)

    def reset(self):
        self.next_id = 1
        self.active  = {}

    def update(self, detections):
        """
        detections : list of (cx, cy, area_px)  — any order
        returns    : list of (drop_id, cx, cy, area_px)  sorted by drop_id
        """
        if not detections:
            self.active = {}
            return []

        # First call — assign IDs in area-descending order so drop 1 is the biggest
        if not self.active:
            detections = sorted(detections, key=lambda d: d[2], reverse=True)
            result = []
            for cx, cy, area in detections:
                self.active[self.next_id] = (cx, cy)
                result.append((self.next_id, cx, cy, area))
                self.next_id += 1
            return result

        # Build all (distance, new_idx, prev_id) pairs
        prev_ids = list(self.active.keys())
        pairs = []
        for ni, (cx, cy, _) in enumerate(detections):
            for pid in prev_ids:
                px, py = self.active[pid]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                pairs.append((dist, ni, pid))
        pairs.sort()

        matched_new  = {}   # new_idx -> drop_id
        used_new     = set()
        used_prev    = set()
        for dist, ni, pid in pairs:
            if ni in used_new or pid in used_prev:
                continue
            if dist > self.max_dist:
                break   # remaining pairs are all farther
            matched_new[ni] = pid
            used_new.add(ni)
            used_prev.add(pid)

        new_active = {}
        result = []
        for ni, (cx, cy, area) in enumerate(detections):
            if ni in matched_new:
                drop_id = matched_new[ni]
            else:
                drop_id = self.next_id
                self.next_id += 1
            new_active[drop_id] = (cx, cy)
            result.append((drop_id, cx, cy, area))

        self.active = new_active
        return sorted(result, key=lambda r: r[0])   # sort by drop_id


# ── detection helpers ────────────────────────────────────────────────────────

def build_background(video_path: str, n_frames: int) -> np.ndarray:
    cap    = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return np.mean(frames, axis=0).astype(np.uint8)


def contour_circularity(c) -> float:
    area      = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def contour_centroid(c):
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])


def detect_contours(gray: np.ndarray, background: np.ndarray,
                    kernel, diff_thresh: int, min_area: int,
                    circ_thresh: float):
    """Return contours passing area + circularity thresholds, sorted area-desc."""
    diff  = cv2.absdiff(gray, background)
    _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    th    = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th    = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if contour_circularity(c) < circ_thresh:
            continue
        out.append(c)
    out.sort(key=cv2.contourArea, reverse=True)
    return out


def is_fully_in_frame(bbox, frame_w, frame_h, margin=3) -> bool:
    x, y, w, h = bbox
    return (x > margin and y > margin
            and x + w < frame_w - margin
            and y + h < frame_h - margin)


# ── per-video analysis ───────────────────────────────────────────────────────

def analyse_video(video_path: str, output_csv: str,
                  bg_frames: int,
                  diff_thresh: int, diff_thresh_post: int,
                  min_area: int,
                  circ_thresh: float, circ_thresh_post: float,
                  margin: int, max_dist: float):

    cap      = cv2.VideoCapture(video_path)
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"\n{'-'*60}")
    print(f"Video : {os.path.basename(video_path)}")
    print(f"Size  : {frame_w}x{frame_h}  |  {n_frames} frames  |  {fps:.1f} fps")

    background = build_background(video_path, bg_frames)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker    = DropletTracker(max_dist=max_dist)

    reference_area      = None
    reference_frame_idx = None
    rows = []

    cap       = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Phase 1: find reference frame ─────────────────────────────────
        if reference_area is None:
            contours = detect_contours(
                gray, background, kernel,
                diff_thresh, min_area, circ_thresh
            )
            # Only accept contours fully inside the frame
            full = [c for c in contours
                    if is_fully_in_frame(cv2.boundingRect(c), frame_w, frame_h, margin)]
            if full:
                reference_area      = sum(cv2.contourArea(c) for c in full)
                reference_frame_idx = frame_idx
                # Initialise tracker with reference contours
                detections = []
                for c in full:
                    ct = contour_centroid(c)
                    if ct:
                        detections.append((ct[0], ct[1], int(cv2.contourArea(c))))
                tracked = tracker.update(detections)
                for drop_id, cx, cy, area in tracked:
                    fully_in = is_fully_in_frame(
                        cv2.boundingRect(full[tracked.index((drop_id, cx, cy, area))]),
                        frame_w, frame_h, margin
                    )
                    rows.append({
                        'frame': frame_idx, 'drop_id': drop_id,
                        'cx': cx, 'cy': cy, 'area_px': area,
                        'percentage': round(area / reference_area * 100, 2),
                        'in_frame': int(fully_in),
                    })
                print(f"  Reference frame {frame_idx}: "
                      f"area = {reference_area:.0f} px²  "
                      f"centroid = ({tracked[0][1]}, {tracked[0][2]})")
            frame_idx += 1
            continue

        # ── Phase 2: post-reference — relaxed thresholds ──────────────────
        contours = detect_contours(
            gray, background, kernel,
            diff_thresh_post, min_area, circ_thresh_post
        )

        if not contours:
            tracker.active = {}   # lost all drops this frame
            frame_idx += 1
            continue

        detections = []
        bbox_map   = {}
        for c in contours:
            ct = contour_centroid(c)
            if ct is None:
                continue
            area = int(cv2.contourArea(c))
            detections.append((ct[0], ct[1], area))
            bbox_map[(ct[0], ct[1], area)] = cv2.boundingRect(c)

        tracked = tracker.update(detections)
        for drop_id, cx, cy, area in tracked:
            bbox     = bbox_map.get((cx, cy, area))
            fully_in = is_fully_in_frame(bbox, frame_w, frame_h, margin) if bbox else False
            rows.append({
                'frame':      frame_idx,
                'drop_id':    drop_id,
                'cx':         cx,
                'cy':         cy,
                'area_px':    area,
                'percentage': round(area / reference_area * 100, 2),
                'in_frame':   int(fully_in),
            })

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  Frame {frame_idx}/{n_frames}")

    cap.release()

    fieldnames = ['frame', 'drop_id', 'cx', 'cy', 'area_px', 'percentage', 'in_frame']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if reference_area is None:
        print("  WARNING: no droplet detected — CSV is empty (headers only)")
    else:
        n_gaps = n_frames - len(set(r['frame'] for r in rows))
        print(f"  Done  |  ref frame {reference_frame_idx}  |  "
              f"{len(rows)} rows  |  {n_gaps} blank frames  ->  {output_csv}")

    return len(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Droplet segmentation — OpenCV background subtraction (v2)'
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--video',  help='Single input video file')
    mode.add_argument('--folder', help='Directory: process every .mp4 found inside')

    p.add_argument('--output',           default=None,
                   help='Output CSV path (single-video mode only; '
                        'default: <video_basename>_results.csv)')
    p.add_argument('--bg_frames',        type=int,   default=30,
                   help='Frames for background model (default: 30)')
    p.add_argument('--diff_thresh',      type=int,   default=25,
                   help='Pixel diff threshold for reference frame detection (default: 25)')
    p.add_argument('--diff_thresh_post', type=int,   default=15,
                   help='Pixel diff threshold post-reference — lower catches spreading lamella (default: 15)')
    p.add_argument('--min_area',         type=int,   default=150,
                   help='Minimum contour area in pixels (default: 150)')
    p.add_argument('--circ_thresh',      type=float, default=0.3,
                   help='Circularity for reference frame detection (default: 0.3)')
    p.add_argument('--circ_thresh_post', type=float, default=0.1,
                   help='Circularity post-reference — lower catches deformed shapes (default: 0.1)')
    p.add_argument('--margin',           type=int,   default=3,
                   help='Edge margin px for fully-in-frame check (default: 3)')
    p.add_argument('--max_dist',         type=float, default=100.0,
                   help='Max centroid distance (px) for ID matching across frames (default: 100)')
    return p.parse_args()


def main():
    args = parse_args()

    kwargs = dict(
        bg_frames        = args.bg_frames,
        diff_thresh      = args.diff_thresh,
        diff_thresh_post = args.diff_thresh_post,
        min_area         = args.min_area,
        circ_thresh      = args.circ_thresh,
        circ_thresh_post = args.circ_thresh_post,
        margin           = args.margin,
        max_dist         = args.max_dist,
    )

    if args.video:
        out = args.output or (Path(args.video).stem + '_results.csv')
        analyse_video(args.video, out, **kwargs)

    else:   # folder mode
        folder = Path(args.folder)
        videos = sorted(folder.glob('*.mp4'))
        if not videos:
            print(f"No .mp4 files found in {folder}")
            return
        print(f"Found {len(videos)} .mp4 files in {folder}")
        total_rows = 0
        for v in videos:
            out = folder / (v.stem + '_results.csv')
            total_rows += analyse_video(str(v), str(out), **kwargs)
        print(f"\n{'='*60}")
        print(f"Batch complete - {len(videos)} videos - {total_rows} total rows")


if __name__ == '__main__':
    main()
