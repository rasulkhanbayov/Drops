"""
VLM Stress Test — Droplet Impact Analysis
==========================================
Tests zero-shot VLMs via OpenRouter on droplet impact frames.
For each test frame we compare VLM estimates against HoughCircles ground truth.

Models tested
-------------
  google/gemini-flash-1.5   — fast / cheap baseline
  openai/gpt-4o-mini        — strong vision, moderate cost

Metrics reported
----------------
  • Centroid MAE (px)              — droplet centre localisation
  • Radius MAE (px)                — size estimation
  • Phase classification accuracy  — falling / spreading / rebounding
  • Spreading width MAE (px)       — contact-width during spreading

Videos tested
-------------
  water.mp4        — pure DI water (simplest, best contrast)
  caonly1.mp4      — CA particles, no surfactant
  cainhsds1.mp4    — CA + high SDS

Usage
-----
    export OPENROUTER_API_KEY=sk-or-v1-...
    python3 vlm_stress_test.py
"""

import os, sys, cv2, json, base64, textwrap, re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────
PX_PER_MM  = 65.625
FPS_ACTUAL = 2996.766489   # from CameraSpecs.jpg (1280×512 resolution)
VIDEOS_DIR = Path("/home/ubuntu/materials/02182026")

SURFACE_ROW = {
    "water.mp4":     433,
    "caonly1.mp4":   399,
    "cainhsds1.mp4": 433,
}
TEST_VIDEOS = list(SURFACE_ROW.keys())

MODELS = [
    "google/gemini-2.0-flash-001",   # $0.10/1M — current Gemini 2.0 Flash
    "openai/gpt-4o-mini",            # $0.15/1M — strong vision baseline
]

N_PRE_IMPACT = 5
N_SPREADING  = 5
N_REBOUND    = 5


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class FrameResult:
    video:     str
    frame_idx: int
    phase:     str   # "falling" | "spreading" | "rebounding"
    model:     str

    gt_cx:          Optional[float] = None
    gt_cy:          Optional[float] = None
    gt_radius:      Optional[float] = None
    gt_spread_width:Optional[float] = None

    vlm_cx:          Optional[float] = None
    vlm_cy:          Optional[float] = None
    vlm_radius:      Optional[float] = None
    vlm_spread_width:Optional[float] = None
    vlm_phase:       Optional[str]   = None
    vlm_raw:         str             = ""
    vlm_confidence:  Optional[str]   = None

    cx_err:      Optional[float] = None
    cy_err:      Optional[float] = None
    radius_err:  Optional[float] = None
    spread_err:  Optional[float] = None
    phase_correct: Optional[bool] = None


# ── Classical CV helpers ─────────────────────────────────────────────────────
def read_frame(video_path: str, fi: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
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
    best = sorted(c, key=lambda x: x[1])[0]      # uppermost = in-flight drop
    return float(best[0]), float(best[1]), float(best[2])


def contact_width(gray: np.ndarray, surface_y: int) -> Optional[float]:
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    band = thresh[max(0, surface_y - 10): surface_y + 5, :]
    cols = np.where(band.max(axis=0) > 0)[0]
    return float(cols[-1] - cols[0]) if len(cols) >= 5 else None


def find_impact_frame(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
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


def find_liftoff_frame(video_path: str, impact_frame: int,
                       surface_y: int, window: int = 60) -> int:
    cap = cv2.VideoCapture(video_path)
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


# ── Frame encoding ────────────────────────────────────────────────────────────
def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


# ── VLM prompt ────────────────────────────────────────────────────────────────
SYSTEM = textwrap.dedent("""\
    You are a precise image-measurement assistant for high-speed shadowgraphy
    videos of water droplets. The image is 1280×512 px. The droplet appears as
    a dark roughly-circular blob on a light background. A horizontal dark line
    near the bottom is the glass surface.

    Respond ONLY with a valid JSON object — no markdown, no prose.
""")

USER_PROMPT = textwrap.dedent("""\
    Analyse this high-speed droplet impact frame.

    1. Classify the droplet phase:
       "falling"    — droplet in free flight above the surface
       "spreading"  — droplet in contact with surface, pancaking outward
       "rebounding" — droplet has left the surface, moving upward

    2. For "falling" or "rebounding":
       Estimate centroid (cx, cy) in pixels and radius in pixels.

    3. For "spreading":
       Estimate the horizontal contact width in pixels (spread_width).
       Also give your best cx, cy, radius of the deformed blob.

    Return exactly:
    {
      "phase": "<falling|spreading|rebounding>",
      "cx": <number or null>,
      "cy": <number or null>,
      "radius": <number or null>,
      "spread_width": <number or null>,
      "confidence": "<low|medium|high>"
    }
""")


def parse_vlm_response(raw: str) -> dict:
    raw = raw.strip()
    # strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def query_vlm(client: OpenAI, model: str, frame: np.ndarray) -> tuple[dict, str]:
    b64 = frame_to_b64(frame)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
    )
    raw = resp.choices[0].message.content or ""
    return parse_vlm_response(raw), raw


# ── Per-video analysis ────────────────────────────────────────────────────────
def analyze_video(video_name: str, client: OpenAI,
                  model: str) -> list[FrameResult]:
    video_path = str(VIDEOS_DIR / video_name)
    surface_y  = SURFACE_ROW[video_name]
    results    = []

    impact_frame  = find_impact_frame(video_path)
    liftoff_frame = find_liftoff_frame(video_path, impact_frame, surface_y)

    phases = {
        "falling":    list(range(max(0, impact_frame - 12),
                                 max(0, impact_frame - 12 + N_PRE_IMPACT))),
        "spreading":  list(range(impact_frame + 2,
                                 impact_frame + 2 + N_SPREADING)),
        "rebounding": list(range(liftoff_frame,
                                 liftoff_frame + N_REBOUND)),
    }

    for phase, frame_idxs in phases.items():
        for fi in frame_idxs:
            frame = read_frame(video_path, fi)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ground truth
            gt_cx = gt_cy = gt_radius = gt_spread = None
            if phase in ("falling", "rebounding"):
                det = hough_detect(gray)
                if det:
                    gt_cx, gt_cy, gt_radius = det
            if phase == "spreading":
                gt_spread = contact_width(gray, surface_y)
                det = hough_detect(gray, min_r=30, max_r=120)
                if det:
                    gt_cx, gt_cy, gt_radius = det

            # VLM
            parsed, raw = query_vlm(client, model, frame)

            r = FrameResult(
                video=video_name, frame_idx=fi, phase=phase, model=model,
                gt_cx=gt_cx, gt_cy=gt_cy, gt_radius=gt_radius,
                gt_spread_width=gt_spread,
                vlm_cx=parsed.get("cx"),
                vlm_cy=parsed.get("cy"),
                vlm_radius=parsed.get("radius"),
                vlm_spread_width=parsed.get("spread_width"),
                vlm_phase=parsed.get("phase"),
                vlm_confidence=parsed.get("confidence"),
                vlm_raw=raw,
            )
            if r.gt_cx    is not None and r.vlm_cx     is not None:
                r.cx_err     = abs(r.vlm_cx     - r.gt_cx)
            if r.gt_cy    is not None and r.vlm_cy     is not None:
                r.cy_err     = abs(r.vlm_cy     - r.gt_cy)
            if r.gt_radius is not None and r.vlm_radius is not None:
                r.radius_err = abs(r.vlm_radius - r.gt_radius)
            if r.gt_spread_width is not None and r.vlm_spread_width is not None:
                r.spread_err = abs(r.vlm_spread_width - r.gt_spread_width)
            if r.vlm_phase is not None:
                r.phase_correct = (r.vlm_phase == phase)

            results.append(r)

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(results: list[FrameResult]) -> dict:
    def mae(vals):
        v = [x for x in vals if x is not None]
        return (round(float(np.mean(v)), 2), len(v)) if v else (float("nan"), 0)

    cx_mae,     n_cx = mae([r.cx_err     for r in results])
    cy_mae,     n_cy = mae([r.cy_err     for r in results])
    r_mae,      n_r  = mae([r.radius_err for r in results])
    sw_mae,     n_sw = mae([r.spread_err for r in results])

    ph = [r.phase_correct for r in results if r.phase_correct is not None]
    ph_acc = round(float(np.mean(ph)) * 100, 1) if ph else float("nan")

    return {
        "cx_mae_px":         cx_mae,  "n_cx":   n_cx,
        "cy_mae_px":         cy_mae,  "n_cy":   n_cy,
        "radius_mae_px":     r_mae,   "n_r":    n_r,
        "spread_mae_px":     sw_mae,  "n_sw":   n_sw,
        "phase_accuracy_pct": ph_acc, "n_phase": len(ph),
        "null_cx":     sum(1 for r in results if r.vlm_cx     is None),
        "null_radius": sum(1 for r in results if r.vlm_radius is None),
        "null_spread": sum(1 for r in results
                           if r.phase == "spreading" and r.vlm_spread_width is None),
        "wrong_phase": sum(1 for r in results
                           if r.phase_correct is not None and not r.phase_correct),
        "cx_mae_mm":     round(cx_mae / PX_PER_MM, 3),
        "cy_mae_mm":     round(cy_mae / PX_PER_MM, 3),
        "radius_mae_mm": round(r_mae  / PX_PER_MM, 3),
        "spread_mae_mm": round(sw_mae / PX_PER_MM, 3),
    }


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(all_results: list[FrameResult],
                 model_metrics: dict[str, dict]):
    print(f"\n{'═'*70}")
    print("  VLM STRESS TEST — RESULTS")
    print(f"{'═'*70}")

    col = 22
    hdr = f"  {'Metric':<28}"
    for m in MODELS:
        short = m.split("/")[1][:col]
        hdr += f"  {short:>{col}}"
    print(hdr)
    print(f"  {'-'*28}" + f"  {'-'*col}" * len(MODELS))

    rows = [
        ("Centroid-X MAE (px)",    "cx_mae_px"),
        ("Centroid-Y MAE (px)",    "cy_mae_px"),
        ("Radius MAE (px)",        "radius_mae_px"),
        ("Spread-width MAE (px)",  "spread_mae_px"),
        ("Phase accuracy (%)",     "phase_accuracy_pct"),
        ("Centroid-X MAE (mm)",    "cx_mae_mm"),
        ("Radius MAE (mm)",        "radius_mae_mm"),
        ("Null cx responses",      "null_cx"),
        ("Null radius responses",  "null_radius"),
        ("Null spread responses",  "null_spread"),
        ("Wrong phase calls",      "wrong_phase"),
    ]
    for label, key in rows:
        line = f"  {label:<28}"
        for m in MODELS:
            v = model_metrics[m].get(key, "—")
            line += f"  {str(v):>{col}}"
        print(line)

    # per-phase centroid MAE
    print(f"\n  Centroid-X MAE by phase (px):")
    for phase in ("falling", "spreading", "rebounding"):
        line = f"    {phase:<12}"
        for m in MODELS:
            sub = [r for r in all_results if r.model == m and r.phase == phase]
            vals = [r.cx_err for r in sub if r.cx_err is not None]
            v = f"{np.mean(vals):.1f} (n={len(vals)})" if vals else "—"
            line += f"  {v:>{col}}"
        print(line)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/droplet-impact-analysis",
            "X-Title": "Droplet Impact VLM Stress Test",
        },
    )

    all_results: list[FrameResult] = []
    model_metrics: dict[str, dict] = {}

    for model in MODELS:
        print(f"\n{'▓'*70}")
        print(f"  MODEL: {model}")
        print(f"{'▓'*70}")
        model_results = []
        for video_name in TEST_VIDEOS:
            surface_y = SURFACE_ROW[video_name]
            video_path = str(VIDEOS_DIR / video_name)
            impact = find_impact_frame(video_path)
            liftoff = find_liftoff_frame(video_path, impact, surface_y)
            print(f"\n  {video_name}: impact={impact}  liftoff={liftoff}")
            results = analyze_video(video_name, client, model)
            for r in results:
                status = f"phase='{r.vlm_phase}'  cx={r.vlm_cx}  r={r.vlm_radius}  conf={r.vlm_confidence}"
                err_cx = f"  err_cx={r.cx_err:.1f}px" if r.cx_err is not None else ""
                print(f"    f{r.frame_idx:04d} [{r.phase:10s}]  {status}{err_cx}")
            model_results.extend(results)
        all_results.extend(model_results)
        model_metrics[model] = compute_metrics(model_results)

    print_report(all_results, model_metrics)

    # save
    out = Path("/home/ubuntu/materials/vlm_stress_test_results.json")
    payload = {
        "model_metrics": model_metrics,
        "frames": [asdict(r) for r in all_results],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    main()
