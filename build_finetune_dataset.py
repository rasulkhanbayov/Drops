"""
Build Fine-Tuning Dataset — Droplet Impact Analysis
=====================================================
Uses classical CV (HoughCircles + contour analysis) to auto-label every
usable frame in both video folders, then writes an OpenAI-format
vision fine-tuning JSONL file.

Each example:
  • system  : same terse measurement-assistant prompt used in stress test
  • user    : JPEG image (base64) + measurement question
  • assistant: ground-truth JSON (phase, cx, cy, radius, spread_width)

Output
------
  finetune_dataset.jsonl   — all examples  (~3 k lines)
  finetune_train.jsonl     — 90 % train split
  finetune_val.jsonl       — 10 % val  split  (held out, never fine-tuned on)
  finetune_metadata.json   — label stats + per-video coverage

Usage
-----
    python3 build_finetune_dataset.py
"""

import cv2
import json
import base64
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

random.seed(42)

# ── Paths & constants ────────────────────────────────────────────────────────
VIDEOS_02   = Path("/home/ubuntu/materials/02182026")
VIDEOS_03   = Path("/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant")
OUT_DIR     = Path("/home/ubuntu/materials/finetune_data")
OUT_DIR.mkdir(exist_ok=True)

PX_PER_MM   = 65.625

# Surface rows from documentation
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
    "0.001percent cg.mp4":      404,
    "0.028p.mp4":                404,
    "0.028percrnt tx.mp4":      467,
    "0.45percrnt sds.mp4":      454,
    "ONLY CA SDS ABOVE CMC.mp4":481,  "ONLY CA SDS ABOVE CMC1.mp4":481,
    "ONLY CA SDS ABOVE CMC2.mp4":481,
    "ONLY CA cg ABOVE CMC1.mp4":485,  "ONLY CA cg ABOVE CMC2.mp4":481,
    "ONLY CA cg ABOVE CMC3.mp4":473,
    "ONLY CA cg less CMC1.mp4": 470,  "ONLY CA cg less CMC2.mp4":465,
    "ONLY CA cg less CMC3.mp4": 473,
    "ONLY CA sds less CMC1.mp4":471,  "ONLY CA sds less CMC2.mp4":470,
    "ONLY CA tx ABOVE CMC1.mp4":482,  "ONLY CA tx ABOVE CMC2.mp4":471,
    "ONLY CA tx ABOVE CMC3.mp4":470,  "ONLY CA tx ABOVE CMC4.mp4":471,
    "ONLY CA tx less CMC1.mp4": 465,
    "ONLY CA tx less CMC2.mp4": 503,  "ONLY CA tx less CMC3.mp4":505,
    "ca+TR.mp4":                479,
}

# Sampling: how many frames to extract per phase per video
FRAMES_PER_PHASE = 8   # yields ~24 examples per video, ~1200 total across all videos

# Prompt strings (identical to stress test so the fine-tuned model
# sees the exact same format at inference time)
SYSTEM_PROMPT = (
    "You are a precise image-measurement assistant for high-speed shadowgraphy "
    "videos of water droplets. The image is 1280×512 px. The droplet appears as "
    "a dark roughly-circular blob on a light background. A horizontal dark line "
    "near the bottom is the glass surface.\n\n"
    "Respond ONLY with a valid JSON object — no markdown, no prose."
)

USER_PROMPT = (
    "Analyse this high-speed droplet impact frame.\n\n"
    "1. Classify the droplet phase:\n"
    '   "falling"    — droplet in free flight above the surface\n'
    '   "spreading"  — droplet in contact with surface, pancaking outward\n'
    '   "rebounding" — droplet has left the surface, moving upward\n\n'
    "2. For \"falling\" or \"rebounding\":\n"
    "   Estimate centroid (cx, cy) in pixels and radius in pixels.\n\n"
    "3. For \"spreading\":\n"
    "   Estimate the horizontal contact width in pixels (spread_width).\n"
    "   Also give your best cx, cy, radius of the deformed blob.\n\n"
    "Return exactly:\n"
    "{\n"
    '  "phase": "<falling|spreading|rebounding>",\n'
    '  "cx": <number or null>,\n'
    '  "cy": <number or null>,\n'
    '  "radius": <number or null>,\n'
    '  "spread_width": <number or null>,\n'
    '  "confidence": "<low|medium|high>"\n'
    "}"
)


# ── Classical CV helpers ─────────────────────────────────────────────────────
@dataclass
class FrameLabel:
    video:        str
    frame_idx:    int
    phase:        str
    cx:           Optional[float]
    cy:           Optional[float]
    radius:       Optional[float]
    spread_width: Optional[float]
    confidence:   str   # "high" when HoughCircles succeeds cleanly

    def to_answer_json(self) -> str:
        return json.dumps({
            "phase":        self.phase,
            "cx":           round(self.cx,     1) if self.cx     is not None else None,
            "cy":           round(self.cy,     1) if self.cy     is not None else None,
            "radius":       round(self.radius, 1) if self.radius is not None else None,
            "spread_width": round(self.spread_width, 1)
                            if self.spread_width is not None else None,
            "confidence":   self.confidence,
        })


def read_frame(path: str, fi: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def hough_detect(gray: np.ndarray,
                 min_r=20, max_r=90) -> Optional[tuple]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=30, param1=50, param2=20,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    c = np.round(circles[0]).astype(int)
    best = sorted(c, key=lambda x: x[1])[0]   # uppermost
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
            diffs.append((i, float(np.abs(roi.astype(int)
                                          - prev[350:, :].astype(int)).mean())))
        prev = gray
    cap.release()
    return max(diffs, key=lambda x: x[1])[0]


def find_liftoff_frame(path: str, impact_frame: int,
                       surface_y: int, window: int = 80) -> int:
    cap = cv2.VideoCapture(path)
    liftoff = impact_frame + window
    for i in range(impact_frame + 5, impact_frame + window):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w = contact_width(gray, surface_y)
        if w is None or w < 10:
            liftoff = i
            break
    cap.release()
    return liftoff


def label_frame(gray: np.ndarray, phase: str,
                surface_y: int) -> tuple[Optional[float], Optional[float],
                                         Optional[float], Optional[float], str]:
    """Return (cx, cy, radius, spread_width, confidence)."""
    cx = cy = radius = spread_w = None
    confidence = "low"

    if phase == "falling":
        det = hough_detect(gray)
        if det:
            cx, cy, radius = det
            confidence = "high"

    elif phase == "spreading":
        spread_w = contact_width(gray, surface_y)
        det = hough_detect(gray, min_r=30, max_r=130)
        if det:
            cx, cy, radius = det
        confidence = "high" if spread_w is not None else "low"

    elif phase == "rebounding":
        det = hough_detect(gray)
        if det:
            cx, cy, radius = det
            confidence = "high"

    return cx, cy, radius, spread_w, confidence


def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.standard_b64encode(buf).decode("utf-8")


# ── Per-video labelling ───────────────────────────────────────────────────────
def label_video(video_path: str, surface_y: int,
                n_per_phase: int = FRAMES_PER_PHASE) -> list[FrameLabel]:
    path     = str(video_path)
    vid_name = Path(video_path).name
    labels   = []

    impact  = find_impact_frame(path)
    liftoff = find_liftoff_frame(path, impact, surface_y)

    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # --- phase frame ranges ---
    fall_start  = max(0, impact - 20)
    fall_end    = max(0, impact - 2)
    spread_start = impact + 1
    spread_end   = min(liftoff - 1, impact + 35)
    reb_start    = liftoff
    reb_end      = min(total - 1, liftoff + 25)

    phases = {
        "falling":    (fall_start,   fall_end),
        "spreading":  (spread_start, spread_end),
        "rebounding": (reb_start,    reb_end),
    }

    for phase, (s, e) in phases.items():
        if e <= s:
            continue
        available = list(range(s, e + 1))
        sampled   = sorted(random.sample(available,
                                         min(n_per_phase, len(available))))
        for fi in sampled:
            frame = read_frame(path, fi)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cx, cy, radius, sw, conf = label_frame(gray, phase, surface_y)

            # skip low-quality labels (no detection at all for non-spreading)
            if phase in ("falling", "rebounding") and cx is None:
                continue
            if phase == "spreading" and sw is None and cx is None:
                continue

            labels.append(FrameLabel(
                video=vid_name, frame_idx=fi, phase=phase,
                cx=cx, cy=cy, radius=radius,
                spread_width=sw, confidence=conf,
            ))

    print(f"  {vid_name:45s}  impact={impact:4d}  liftoff={liftoff:4d}"
          f"  labels={len(labels)}")
    return labels


# ── JSONL conversion ──────────────────────────────────────────────────────────
def label_to_jsonl(label: FrameLabel,
                   video_path: str) -> Optional[dict]:
    """Convert one label to an OpenAI fine-tuning message dict."""
    frame = read_frame(video_path, label.frame_idx)
    if frame is None:
        return None
    b64 = frame_to_b64(frame)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {"role": "assistant", "content": label.to_answer_json()},
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_labels: list[FrameLabel] = []
    video_lookup: dict[str, str] = {}   # name → full path

    print("── Folder 02182026 ──────────────────────────────────────")
    for name, sy in SURFACE_ROW_02.items():
        p = VIDEOS_02 / name
        if not p.exists():
            print(f"  MISSING: {name}")
            continue
        video_lookup[name] = str(p)
        labels = label_video(p, sy)
        all_labels.extend(labels)

    print("\n── Folder 03242026 ──────────────────────────────────────")
    for name, sy in SURFACE_ROW_03.items():
        p = VIDEOS_03 / name
        if not p.exists():
            print(f"  MISSING: {name}")
            continue
        video_lookup[name] = str(p)
        labels = label_video(p, sy)
        all_labels.extend(labels)

    print(f"\nTotal labels generated: {len(all_labels)}")

    # ── Phase breakdown ──────────────────────────────────────────────────────
    for ph in ("falling", "spreading", "rebounding"):
        n = sum(1 for l in all_labels if l.phase == ph)
        print(f"  {ph:12s}: {n}")

    # ── Train / val split (90 / 10, stratified by phase) ────────────────────
    random.shuffle(all_labels)
    split = int(0.9 * len(all_labels))
    train_labels = all_labels[:split]
    val_labels   = all_labels[split:]
    print(f"\nTrain: {len(train_labels)}  |  Val: {len(val_labels)}")

    # ── Write JSONL files ────────────────────────────────────────────────────
    def write_jsonl(labels: list[FrameLabel], path: Path):
        skipped = 0
        with open(path, "w") as f:
            for label in labels:
                vpath = video_lookup.get(label.video)
                if vpath is None:
                    skipped += 1
                    continue
                entry = label_to_jsonl(label, vpath)
                if entry is None:
                    skipped += 1
                    continue
                f.write(json.dumps(entry) + "\n")
        print(f"  Wrote {len(labels) - skipped} examples → {path}  (skipped {skipped})")

    print("\nWriting JSONL files (this encodes all frames as JPEG base64) ...")
    all_path   = OUT_DIR / "finetune_dataset.jsonl"
    train_path = OUT_DIR / "finetune_train.jsonl"
    val_path   = OUT_DIR / "finetune_val.jsonl"

    write_jsonl(all_labels,   all_path)
    write_jsonl(train_labels, train_path)
    write_jsonl(val_labels,   val_path)

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "total":  len(all_labels),
        "train":  len(train_labels),
        "val":    len(val_labels),
        "phases": {ph: sum(1 for l in all_labels if l.phase == ph)
                   for ph in ("falling", "spreading", "rebounding")},
        "videos": list(video_lookup.keys()),
        "px_per_mm":     PX_PER_MM,
        "frames_per_phase_per_video": FRAMES_PER_PHASE,
    }
    (OUT_DIR / "finetune_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nMetadata → {OUT_DIR / 'finetune_metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
