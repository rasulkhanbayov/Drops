"""
benchmark_eval.py
-----------------
Evaluates any vision-language model on the PhysDrop benchmark.

Usage
-----
    # Zero-shot evaluation via OpenRouter
    export OPENROUTER_API_KEY=sk-or-v1-...
    python3 benchmark_eval.py --model google/gemini-2.0-flash-001

    # Multiple models
    python3 benchmark_eval.py --model openai/gpt-4o --model google/gemini-2.0-flash-001

    # Limit to Task 6 only, skip already-evaluated entries
    python3 benchmark_eval.py --model openai/gpt-4o --task task6 --resume

    # Use local fine-tuned Qwen model (no API key needed)
    python3 benchmark_eval.py --model local/qwen25vl-finetuned --local

Supported models (via OpenRouter)
----------------------------------
    google/gemini-2.0-flash-001
    google/gemini-2.5-flash-preview
    openai/gpt-4o
    openai/gpt-4o-mini
    anthropic/claude-3-5-sonnet
    anthropic/claude-3-7-sonnet
    qwen/qwen2.5-vl-72b-instruct
    meta-llama/llama-3.2-90b-vision-instruct

Output
------
    benchmark/results/<model_slug>_results.json   raw per-entry results
    benchmark/results/<model_slug>_metrics.json   aggregated metrics

Metrics computed
----------------
  Task 1/4:
    phase_accuracy_pct
    cx_mae_px, cy_mae_px, radius_mae_px   (falling + rebounding frames)
    spread_mae_px                          (spreading frames)
    null_rate_pct                          (refused / unparseable responses)

  Task 6:
    accuracy_pct                           (4-class fluid classification)
    per_class_accuracy                     {A:%, B:%, C:%, D:%}
    confusion_matrix                       4×4
    null_rate_pct

  Physics accuracy (Task 1/4):
    beta_max_mae    mean |VLM-derived β_max − CV β_max|  (requires D0 + spread_width)
"""

import os, sys, re, json, time, base64, argparse
import numpy as np
from pathlib import Path
from typing import Optional
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/home/ubuntu/materials")
BENCH_DIR = BASE / "benchmark"
RES_DIR   = BENCH_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = BENCH_DIR / "benchmark.json"
DEFAULT_PX_PER_MM = 65.625

# ── Model routing ─────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
LOCAL_QWEN_DIR  = BASE / "finetune_data" / "qwen25vl_lora"

def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def image_content_item(img_path: Path) -> dict:
    b64 = load_image_b64(img_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


# ── API call — OpenRouter ─────────────────────────────────────────────────────

def call_openrouter(
    client: OpenAI,
    model: str,
    prompt: str,
    images: list[Path],
    max_tokens: int = 300,
    retries: int = 3,
) -> str:
    content = []
    for img in images:
        content.append(image_content_item(img))
    content.append({"type": "text", "text": prompt})

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{retries}] {e} — waiting {wait}s")
            time.sleep(wait)
    return ""


# ── API call — local Qwen2.5-VL fine-tuned ───────────────────────────────────

_qwen_model = None
_qwen_proc  = None


def _load_local_qwen():
    global _qwen_model, _qwen_proc
    if _qwen_model is not None:
        return
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel
    import torch

    base_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter  = str(LOCAL_QWEN_DIR / "final_adapter")
    print(f"  Loading local Qwen base model...")
    proc  = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Loading LoRA adapter from {adapter} ...")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    _qwen_model, _qwen_proc = model, proc
    print("  Local model ready.")


def call_local_qwen(prompt: str, images: list[Path], max_tokens: int = 300) -> str:
    import torch
    from PIL import Image as PILImage
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import process_vision_info

    _load_local_qwen()

    pil_images = [PILImage.open(p).convert("RGB") for p in images]
    content = []
    for img in pil_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = _qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _qwen_proc(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(_qwen_model.device)
    with torch.no_grad():
        out_ids = _qwen_model.generate(**inputs, max_new_tokens=max_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return _qwen_proc.batch_decode(trimmed, skip_special_tokens=True)[0]


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_task14(raw: str) -> dict:
    """Extract JSON fields from VLM response for Task 1/4."""
    raw = raw.strip()
    # strip markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def parse_task6(raw: str) -> dict:
    """Extract choice (A/B/C/D) from VLM response for Task 6."""
    raw_clean = raw.strip()
    raw_clean = re.sub(r"^```[a-z]*\n?", "", raw_clean)
    raw_clean = re.sub(r"\n?```$", "", raw_clean)
    m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            return obj
        except Exception:
            pass
    # fallback: look for a bare letter
    m2 = re.search(r"\b([ABCD])\b", raw)
    if m2:
        return {"choice": m2.group(1)}
    return {}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    task14 = [r for r in results if r["task"] == "task1_phase_and_measurement"]
    task6  = [r for r in results if r["task"] == "task6_fluid_classification"]

    metrics = {}

    # ── Task 1/4 ──────────────────────────────────────────────────────────────
    if task14:
        # filter to entries with known fluid class (exclude scale videos)
        task14 = [r for r in task14 if r.get("fluid_class") not in (None, "unknown")]

        total      = len(task14)
        phase_ok   = sum(1 for r in task14 if r.get("phase_correct"))
        null_count = sum(1 for r in task14 if r.get("vlm_phase") is None)

        # position + size errors (falling/rebounding only, where GT is available)
        cx_errs, cy_errs, r_errs, sw_errs = [], [], [], []
        for r in task14:
            gt_phase = r.get("gt_phase")
            if gt_phase in ("falling", "rebounding"):
                if r.get("gt_cx") is not None and r.get("vlm_cx") is not None:
                    cx_errs.append(abs(r["vlm_cx"] - r["gt_cx"]))
                if r.get("gt_cy") is not None and r.get("vlm_cy") is not None:
                    cy_errs.append(abs(r["vlm_cy"] - r["gt_cy"]))
                if r.get("gt_radius") is not None and r.get("vlm_radius") is not None:
                    r_errs.append(abs(r["vlm_radius"] - r["gt_radius"]))
            elif gt_phase == "spreading":
                if r.get("gt_spread") is not None and r.get("vlm_spread") is not None:
                    sw_errs.append(abs(r["vlm_spread"] - r["gt_spread"]))

        def _mean(errs):
            return round(np.mean(errs), 2) if errs else None

        # Per-entry px→mm errors using the correct scale for each folder
        cx_errs_mm, r_errs_mm, sw_errs_mm = [], [], []
        for r in task14:
            ppm = r.get("px_per_mm", DEFAULT_PX_PER_MM)
            gt_phase = r.get("gt_phase")
            if gt_phase in ("falling", "rebounding"):
                if r.get("gt_cx") is not None and r.get("vlm_cx") is not None:
                    cx_errs_mm.append(abs(r["vlm_cx"] - r["gt_cx"]) / ppm)
                if r.get("gt_radius") is not None and r.get("vlm_radius") is not None:
                    r_errs_mm.append(abs(r["vlm_radius"] - r["gt_radius"]) / ppm)
            elif gt_phase == "spreading":
                if r.get("gt_spread") is not None and r.get("vlm_spread") is not None:
                    sw_errs_mm.append(abs(r["vlm_spread"] - r["gt_spread"]) / ppm)

        metrics["task14"] = {
            "n_total":              total,
            "phase_accuracy_pct":   round(100 * phase_ok / total, 1) if total else None,
            "null_rate_pct":        round(100 * null_count / total, 1) if total else None,
            "cx_mae_px":            _mean(cx_errs),
            "cy_mae_px":            _mean(cy_errs),
            "radius_mae_px":        _mean(r_errs),
            "spread_mae_px":        _mean(sw_errs),
            "cx_mae_mm":            round(_mean(cx_errs_mm), 3) if cx_errs_mm else None,
            "radius_mae_mm":        round(_mean(r_errs_mm),  3) if r_errs_mm  else None,
            "spread_mae_mm":        round(_mean(sw_errs_mm), 3) if sw_errs_mm else None,
        }

        # Physics accuracy: VLM-derived β_max vs classical CV β_max
        beta_errs = []
        for r in task14:
            cv_beta  = r.get("gt_beta_max")
            vd0      = r.get("vlm_radius")          # px → D0 = 2 * radius
            vsw      = r.get("vlm_spread")           # px spread width at max
            if cv_beta and vd0 and vsw and vd0 > 0:
                vlm_beta = vsw / (2 * vd0)           # β = D_max / D0
                beta_errs.append(abs(vlm_beta - cv_beta))
        metrics["task14"]["beta_max_vlm_mae"] = _mean(beta_errs)

    # ── Task 6 ────────────────────────────────────────────────────────────────
    if task6:
        total      = len(task6)
        null_count = sum(1 for r in task6 if not r.get("vlm_choice"))
        correct    = sum(1 for r in task6 if r.get("correct"))
        choices    = ["A", "B", "C", "D"]

        per_class = {}
        conf_mat  = {gt: {pr: 0 for pr in choices} for gt in choices}
        for r in task6:
            gt = r.get("gt_choice")
            pr = r.get("vlm_choice")
            if not gt: continue
            if pr in choices:
                conf_mat[gt][pr] += 1
            n_gt = sum(1 for x in task6 if x.get("gt_choice") == gt)
            n_ok = sum(1 for x in task6
                       if x.get("gt_choice") == gt and x.get("correct"))
            per_class[gt] = round(100 * n_ok / n_gt, 1) if n_gt else None

        metrics["task6"] = {
            "n_total":          total,
            "accuracy_pct":     round(100 * correct / total, 1) if total else None,
            "null_rate_pct":    round(100 * null_count / total, 1) if total else None,
            "per_class_accuracy": per_class,
            "confusion_matrix": conf_mat,
            "chance_level_pct": 25.0,
        }

    return metrics


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    model_id:   str,
    tasks:      list[str],
    use_local:  bool,
    resume:     bool,
    delay:      float,
    max_entries: Optional[int],
):
    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    slug = model_slug(model_id)
    res_path = RES_DIR / f"{slug}_results.json"

    # Load previous results for resume
    done_ids = set()
    prev_results = []
    if resume and res_path.exists():
        with open(res_path) as f:
            prev_results = json.load(f)
        done_ids = {r["id"] for r in prev_results}
        print(f"Resuming: {len(done_ids)} entries already done.")

    # Set up API client
    client = None
    if not use_local:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            sys.exit("Error: OPENROUTER_API_KEY not set. Export it or use --local.")
        client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)

    results = list(prev_results)
    todo = [
        e for e in benchmark
        if e["id"] not in done_ids
        and (not tasks or any(t in e["task"] for t in tasks))
        and e.get("fluid_class") not in (None, "unknown")  # skip scale videos
    ]
    if max_entries:
        todo = todo[:max_entries]

    print(f"Model : {model_id}")
    print(f"Tasks : {tasks or 'all'}")
    print(f"Entries to evaluate: {len(todo)}")
    print()

    for i, entry in enumerate(todo):
        task  = entry["task"]
        eid   = entry["id"]
        video = entry["video"]
        imgs  = [BASE / f["path"] for f in entry["frames"] if (BASE / f["path"]).exists()]

        if not imgs:
            print(f"  [{i+1}/{len(todo)}] SKIP (no frames): {eid}")
            continue

        print(f"  [{i+1}/{len(todo)}] {task[:6]}  {video:40s}", end="", flush=True)

        # Call model
        prompt = entry["prompt"]
        if use_local:
            raw = call_local_qwen(prompt, imgs)
        else:
            raw = call_openrouter(client, model_id, prompt, imgs)

        # Parse response
        if "task6" in task:
            parsed = parse_task6(raw)
            gt_choice = entry["gt"].get("fluid_choice")
            vlm_choice = parsed.get("choice", "").upper().strip()
            if vlm_choice not in ("A","B","C","D"):
                vlm_choice = None
            result = {
                "id":         eid,
                "task":       task,
                "video":      video,
                "folder":     entry["folder"],
                "fluid_class": entry["fluid_class"],
                "model":      model_id,
                "gt_choice":  gt_choice,
                "vlm_choice": vlm_choice,
                "vlm_reasoning": parsed.get("reasoning"),
                "vlm_confidence": parsed.get("confidence"),
                "vlm_raw":    raw,
                "correct":    (vlm_choice == gt_choice) if vlm_choice else False,
                "gt_beta_max": entry["gt"].get("beta_max"),
                "gt_COR":      entry["gt"].get("COR"),
            }
            status = f"GT={gt_choice} VLM={vlm_choice} {'✓' if result['correct'] else '✗'}"

        else:  # task1/4
            parsed = parse_task14(raw)
            gt     = entry["gt"]
            gt_phase  = gt.get("phase")
            vlm_phase = parsed.get("phase")

            def _f(d, k): return float(d[k]) if d.get(k) is not None else None

            result = {
                "id":           eid,
                "task":         task,
                "video":        video,
                "folder":       entry["folder"],
                "fluid_class":  entry["fluid_class"],
                "px_per_mm":    entry.get("px_per_mm", DEFAULT_PX_PER_MM),
                "model":        model_id,
                "gt_phase":     gt_phase,
                "gt_cx":        gt.get("cx_px"),
                "gt_cy":        gt.get("cy_px"),
                "gt_radius":    gt.get("radius_px"),
                "gt_spread":    gt.get("spread_px"),
                "gt_beta_max":  gt.get("beta_max"),
                "gt_COR":       gt.get("COR"),
                "vlm_phase":    vlm_phase,
                "vlm_cx":       _f(parsed, "cx"),
                "vlm_cy":       _f(parsed, "cy"),
                "vlm_radius":   _f(parsed, "radius"),
                "vlm_spread":   _f(parsed, "spread_width"),
                "vlm_confidence": parsed.get("confidence"),
                "vlm_raw":      raw,
                "phase_correct": (vlm_phase == gt_phase) if vlm_phase else False,
            }
            # Compute errors
            for field, gt_key, vlm_key in [
                ("cx_err",     "gt_cx",     "vlm_cx"),
                ("cy_err",     "gt_cy",     "vlm_cy"),
                ("radius_err", "gt_radius", "vlm_radius"),
                ("spread_err", "gt_spread", "vlm_spread"),
            ]:
                g, v = result.get(gt_key), result.get(vlm_key)
                result[field] = round(abs(v - g), 1) if (g is not None and v is not None) else None

            ph_ok = "✓" if result["phase_correct"] else "✗"
            r_err = result.get("radius_err")
            status = f"phase={vlm_phase}({ph_ok}) r_err={r_err}"

        results.append(result)
        print(f"  {status}")

        # Save incrementally
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

        if delay > 0:
            time.sleep(delay)

    # ── Final metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics(results)
    metrics["model"]       = model_id
    metrics["n_evaluated"] = len(results)

    met_path = RES_DIR / f"{slug}_metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    if "task14" in metrics:
        m = metrics["task14"]
        print(f"Task 1/4  phase_acc={m['phase_accuracy_pct']}%  "
              f"radius_MAE={m['radius_mae_mm']}mm  "
              f"spread_MAE={m['spread_mae_mm']}mm  "
              f"null={m['null_rate_pct']}%")
    if "task6" in metrics:
        m = metrics["task6"]
        print(f"Task 6    accuracy={m['accuracy_pct']}%  "
              f"(chance=25%)  null={m['null_rate_pct']}%")
        print(f"          per-class: {m['per_class_accuracy']}")
    print(f"Results → {res_path}")
    print(f"Metrics → {met_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PhysDrop benchmark evaluator")
    parser.add_argument("--model",   action="append", default=[], dest="models",
                        help="Model ID (repeat for multiple). Default: gemini-2.0-flash")
    parser.add_argument("--task",    action="append", default=[], dest="tasks",
                        help="Filter tasks: 'task6', 'task1'. Default: all.")
    parser.add_argument("--local",   action="store_true",
                        help="Use local fine-tuned Qwen model (no API key needed)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip entries already in results file")
    parser.add_argument("--delay",   type=float, default=0.3,
                        help="Seconds between API calls (default 0.3)")
    parser.add_argument("--max",     type=int, default=None,
                        help="Max entries to evaluate (for quick tests)")
    args = parser.parse_args()

    models = args.models or ["google/gemini-2.0-flash-001"]

    for model_id in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*60}")
        if args.local:
            model_id = "local/qwen25vl-finetuned"
        evaluate(
            model_id    = model_id,
            tasks       = args.tasks,
            use_local   = args.local,
            resume      = args.resume,
            delay       = args.delay,
            max_entries = args.max,
        )


if __name__ == "__main__":
    main()
