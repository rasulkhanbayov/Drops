"""
Evaluate Fine-Tuned Qwen2.5-VL vs Zero-Shot Baseline
=====================================================
Runs the same 45-frame test as vlm_stress_test.py but uses the
local fine-tuned LoRA adapter instead of calling OpenRouter.

Reports the same metrics so the two tables can be compared directly.

Usage
-----
    /opt/anaconda3/2024.02-1/conda_envs/ml_dl_gpu_base/bin/python \
        eval_finetuned.py
"""

import cv2
import json
import re
import base64
import io
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ── Paths ──────────────────────────────────────────────────────────────────────
ADAPTER_PATH = Path("/home/ubuntu/materials/finetune_data/qwen25vl_lora/final_adapter")
VIDEOS_DIR   = Path("/home/ubuntu/materials/02182026")

PX_PER_MM  = 65.625
FPS_ACTUAL = 2996.766489   # from CameraSpecs.jpg (1280×512 resolution)

SURFACE_ROW = {
    "water.mp4":     433,
    "caonly1.mp4":   399,
    "cainhsds1.mp4": 433,
}
TEST_VIDEOS = list(SURFACE_ROW.keys())
N_PER_PHASE = 5

# ── Same prompts as stress test ───────────────────────────────────────────────
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


# ── Classical CV helpers (copied from stress test) ─────────────────────────────
@dataclass
class FrameResult:
    video: str; frame_idx: int; phase: str
    gt_cx: Optional[float] = None;   gt_cy: Optional[float] = None
    gt_radius: Optional[float] = None; gt_spread_width: Optional[float] = None
    vlm_cx: Optional[float] = None;  vlm_cy: Optional[float] = None
    vlm_radius: Optional[float] = None; vlm_spread_width: Optional[float] = None
    vlm_phase: Optional[str] = None; vlm_confidence: Optional[str] = None
    vlm_raw: str = ""
    cx_err: Optional[float] = None;  cy_err: Optional[float] = None
    radius_err: Optional[float] = None; spread_err: Optional[float] = None
    phase_correct: Optional[bool] = None


def read_frame(path: str, fi: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def hough_detect(gray, min_r=20, max_r=90):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=30, param1=50, param2=20,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    c = np.round(circles[0]).astype(int)
    best = sorted(c, key=lambda x: x[1])[0]
    return float(best[0]), float(best[1]), float(best[2])


def contact_width(gray, surface_y):
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


def find_liftoff_frame(path: str, impact: int, surface_y: int, window=60) -> int:
    cap = cv2.VideoCapture(path)
    liftoff = impact + window
    for i in range(impact + 5, impact + window):
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


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ── VLM inference ─────────────────────────────────────────────────────────────
def query_finetuned(model, processor, frame_bgr: np.ndarray) -> tuple[dict, str]:
    # BGR → PIL RGB
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image",  "image": pil},
                {"type": "text",   "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[pil],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the new tokens
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_ids[0], skip_special_tokens=True)
    return parse_response(raw), raw


# ── Evaluation loop ────────────────────────────────────────────────────────────
def evaluate(model, processor) -> list[FrameResult]:
    all_results = []

    for video_name in TEST_VIDEOS:
        video_path = str(VIDEOS_DIR / video_name)
        surface_y  = SURFACE_ROW[video_name]

        impact  = find_impact_frame(video_path)
        liftoff = find_liftoff_frame(video_path, impact, surface_y)

        phases = {
            "falling":    list(range(max(0, impact - 12),
                                     max(0, impact - 12 + N_PER_PHASE))),
            "spreading":  list(range(impact + 2, impact + 2 + N_PER_PHASE)),
            "rebounding": list(range(liftoff, liftoff + N_PER_PHASE)),
        }

        print(f"\n  {video_name}  (impact={impact}, liftoff={liftoff})")

        for phase, frame_idxs in phases.items():
            for fi in frame_idxs:
                frame = read_frame(video_path, fi)
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gt_cx = gt_cy = gt_r = gt_sw = None
                if phase in ("falling", "rebounding"):
                    det = hough_detect(gray)
                    if det:
                        gt_cx, gt_cy, gt_r = det
                if phase == "spreading":
                    gt_sw = contact_width(gray, surface_y)
                    det = hough_detect(gray, min_r=30, max_r=120)
                    if det:
                        gt_cx, gt_cy, gt_r = det

                parsed, raw = query_finetuned(model, processor, frame)

                r = FrameResult(
                    video=video_name, frame_idx=fi, phase=phase,
                    gt_cx=gt_cx, gt_cy=gt_cy, gt_radius=gt_r,
                    gt_spread_width=gt_sw,
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

                status = (f"phase='{r.vlm_phase}'  cx={r.vlm_cx}  r={r.vlm_radius}"
                          f"  sw={r.vlm_spread_width}  conf={r.vlm_confidence}")
                err_cx = f"  err_cx={r.cx_err:.1f}px" if r.cx_err is not None else ""
                print(f"    f{fi:04d} [{phase:10s}]  {status}{err_cx}")

                all_results.append(r)

    return all_results


def report(results: list[FrameResult]):
    def mae(vals):
        v = [x for x in vals if x is not None]
        return (round(float(np.mean(v)), 2), len(v)) if v else (float("nan"), 0)

    cx_mae,  n_cx = mae([r.cx_err     for r in results])
    cy_mae,  n_cy = mae([r.cy_err     for r in results])
    r_mae,   n_r  = mae([r.radius_err for r in results])
    sw_mae,  n_sw = mae([r.spread_err for r in results])
    ph = [r.phase_correct for r in results if r.phase_correct is not None]
    ph_acc = round(float(np.mean(ph)) * 100, 1) if ph else float("nan")

    print(f"\n{'═'*60}")
    print("  FINE-TUNED MODEL — EVALUATION RESULTS")
    print(f"{'═'*60}")
    print(f"  Frames evaluated       : {len(results)}")
    print(f"  Centroid-X MAE (px)    : {cx_mae}  (n={n_cx})")
    print(f"  Centroid-Y MAE (px)    : {cy_mae}  (n={n_cy})")
    print(f"  Radius MAE (px)        : {r_mae}  (n={n_r})")
    print(f"  Spread-width MAE (px)  : {sw_mae}  (n={n_sw})")
    print(f"  Phase accuracy         : {ph_acc}%  (n={len(ph)})")
    print(f"\n  In physical units:")
    print(f"  Centroid-X MAE (mm)    : {round(cx_mae / PX_PER_MM, 3)}")
    print(f"  Radius MAE (mm)        : {round(r_mae  / PX_PER_MM, 3)}")
    print(f"  Spread-width MAE (mm)  : {round(sw_mae / PX_PER_MM, 3)}")

    print(f"\n  Failure modes:")
    print(f"    Null cx              : {sum(1 for r in results if r.vlm_cx is None)}")
    print(f"    Null radius          : {sum(1 for r in results if r.vlm_radius is None)}")
    print(f"    Null spread          : {sum(1 for r in results if r.phase=='spreading' and r.vlm_spread_width is None)}")
    print(f"    Wrong phase          : {sum(1 for r in results if r.phase_correct is not None and not r.phase_correct)}")

    # ── Zero-shot comparison (from saved stress test results) ─────────────────
    st_path = Path("/home/ubuntu/materials/vlm_stress_test_results.json")
    if st_path.exists():
        st = json.loads(st_path.read_text())
        print(f"\n{'─'*60}")
        print("  COMPARISON: zero-shot vs fine-tuned")
        print(f"{'─'*60}")
        print(f"  {'Metric':<28}  {'Gemini (0-shot)':>16}  {'GPT-4o-mini (0-shot)':>20}  {'Qwen FT':>10}")
        metrics = st["model_metrics"]
        g = metrics.get("google/gemini-2.0-flash-001", {})
        p = metrics.get("openai/gpt-4o-mini", {})
        ft = {
            "cx_mae_px": cx_mae, "cy_mae_px": cy_mae,
            "radius_mae_px": r_mae, "spread_mae_px": sw_mae,
            "phase_accuracy_pct": ph_acc,
        }
        rows = [
            ("Centroid-X MAE (px)",   "cx_mae_px"),
            ("Centroid-Y MAE (px)",   "cy_mae_px"),
            ("Radius MAE (px)",       "radius_mae_px"),
            ("Spread-width MAE (px)", "spread_mae_px"),
            ("Phase accuracy (%)",    "phase_accuracy_pct"),
        ]
        for label, key in rows:
            gv = g.get(key, "—")
            pv = p.get(key, "—")
            fv = ft.get(key, "—")
            print(f"  {label:<28}  {str(gv):>16}  {str(pv):>20}  {str(fv):>10}")

    out = Path("/home/ubuntu/materials/eval_finetuned_results.json")
    out.write_text(json.dumps({
        "metrics": {
            "cx_mae_px": cx_mae, "cy_mae_px": cy_mae,
            "radius_mae_px": r_mae, "spread_mae_px": sw_mae,
            "phase_accuracy_pct": ph_acc,
        },
        "frames": [asdict(r) for r in results],
    }, indent=2))
    print(f"\n  Detailed results → {out}")


def main():
    print(f"Loading fine-tuned model from {ADAPTER_PATH}")

    # Load base model
    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(
        str(ADAPTER_PATH), trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    print("Model ready.")

    results = evaluate(model, processor)
    report(results)


if __name__ == "__main__":
    main()
