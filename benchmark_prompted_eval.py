"""
benchmark_prompted_eval.py
--------------------------
Runs the 6 representative videos from supervisor_results_preview.md through
domain-engineered comprehensive prompts (vs. the zero-shot prompts already run).

Saves results to benchmark/results/prompted_<model_slug>_results.json
and prints a comparison table at the end.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python3 benchmark_prompted_eval.py --models anthropic/claude-sonnet-4-5 openai/gpt-4o
"""

import os, sys, re, json, time, base64, argparse
import numpy as np
from pathlib import Path

BASE      = Path("/home/ubuntu/materials")
BENCH_DIR = BASE / "benchmark"
RES_DIR   = BENCH_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = BENCH_DIR / "benchmark.json"

TARGET_VIDEOS = [
    "water2.mp4",
    "cainhcg1.mp4",
    "ONLY CA sds less CMC1.mp4",
    "0.45percrnt sds.mp4",
    "cainhcg 0.08.mp4",
    "0.028tx.mp4",
]

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── Domain-engineered prompts ─────────────────────────────────────────────────

TASK6_PROMPTED = """\
CONTEXT — PHYSICS AND IMAGING
You are analysing high-speed shadowgraphy footage of droplet impact on a superhydrophobic glass surface.

Imaging conditions:
• Camera: Photron high-speed, 2997 fps, resolution 1280×512 px
• Calibration: ~65.6 px per mm (droplet diameter ≈ 2 mm ≈ 130 px)
• Drop height: 6.5 cm → impact velocity ≈ 1.1 m/s
• Weber number We = ρ U² D₀ / σ. For pure water at 1.1 m/s and D₀ = 2 mm: We ≈ 33.
  Surfactant lowers σ, raising We; nanoparticles alter spreading and rebound rheology.
• The surface is superhydrophobic (contact angle > 150°). All fluids fully rebound in the
  absence of wetting agents strong enough to overcome the Cassie-Baxter state.

Image appearance:
• Droplet: dark circular blob (shadowgraphy — back-lit, droplet blocks light)
• Caustic ring: bright halo around the falling/rebounding droplet (lensing artefact — NOT a second droplet)
• Surface: horizontal dark line ~60% down the frame
• Background: uniform bright grey; no colour information

THREE FRAMES ARE PROVIDED:
  Frame 1 — droplet in free fall, just before impact
  Frame 2 — moment of maximum lateral spreading (widest contact footprint)
  Frame 3 — droplet during or after rebound, leaving the surface

VISUAL CUES FOR EACH FLUID CLASS — what to look for:

A) Pure water (control)
   • Clean spherical shape before and after impact
   • Moderate spreading (β_max = D_spread/D₀ ≈ 2.0–2.5)
   • Complete, clean rebound — droplet detaches fully as a sphere
   • No satellite droplets; no residue on surface

B) Surfactant solution only (SDS/TX-100/CG, above CMC)
   • Spreading is WIDER than pure water (lower surface tension, β_max ≈ 2.5–3.5)
   • Rebound may be partial or suppressed — surfactant promotes wetting
   • Droplet may leave a thin liquid film / residue on surface after rebound
   • Rebound droplet is smaller (satellite droplets possible)
   • Shape on rebound: more flattened, less spherical than pure water

C) CA nanoparticles + surfactant (surfactant still present in droplet)
   • Similar or slightly wider spread than surfactant-only (nanoparticles add viscosity)
   • Rebound is notably suppressed or very sluggish compared to B
   • Droplet surface appears rougher / irregular during spreading and rebound
   • May show internal structure or asymmetry during spreading (particle jamming)
   • Least elastic rebound among the four classes

D) Washed CA nanoparticles (no free surfactant — nanoparticles only in pure water base)
   • Spreading similar to pure water (no surfactant to lower σ), β_max ≈ 2.0–2.5
   • Rebound is clean but may be slightly slower than pure water due to particle drag
   • Droplet shape before/after similar to water; the key distinguishing feature is
     that behaviour looks nearly identical to pure water — subtle differences only
   • No film left on surface (no surfactant)

DECISION STRATEGY:
1. Compare spreading width in Frame 2 to approximate droplet diameter from Frame 1.
   If β_max < 2.5 → likely A or D. If β_max > 2.5 → likely B or C.
2. Examine the rebound (Frame 3): full clean sphere → A or D; suppressed/partial → B or C.
3. If spread is large AND rebound is suppressed → C (nanoparticles + surfactant).
4. If spread is large AND rebound occurs but droplet is smaller → B (surfactant only).
5. A vs D are the hardest to distinguish visually; D may show very slight roughness.

Respond ONLY with JSON — no prose outside the JSON block:
{"choice":"A|B|C|D","reasoning":"two sentences citing specific visual evidence","confidence":"high|medium|low"}
"""

# Per-video calibration constants (for the 6 target videos, all in 02182026 / 03242026)
PX_PER_MM_DICT = {
    "water2.mp4":              65.625,
    "cainhcg1.mp4":            65.625,
    "cainhcg 0.08.mp4":        65.625,
    "0.45percrnt sds.mp4":     65.625,
    "ONLY CA sds less CMC1.mp4": 65.625,
    "0.028tx.mp4":             65.625,
}

TASK14_PROMPTED_TEMPLATE = """\
CONTEXT — HIGH-SPEED SHADOWGRAPHY FRAME ANALYSIS
Camera: Photron high-speed, 2997 fps, 1280×512 px.
Calibration: {px_per_mm:.1f} px per mm ({mm_per_px:.4f} mm/px).
Drop height: 6.5 cm → impact velocity ≈ 1.1 m/s.

IMAGE APPEARANCE:
• Droplet diameter D₀ ≈ 2 mm ≈ {d0_px:.0f} px — use this as a sanity check for your size estimate
• Droplet: dark circular or oval blob (back-lit shadowgraphy — droplet blocks light)
• Caustic ring: bright halo just outside the droplet perimeter — this is a lens artefact,
  do NOT confuse it with the droplet boundary. The true droplet edge is where the dark region ends.
• Surface: horizontal dark line at approximately y ≈ {surface_row} px from the top
• Background: uniform bright grey; pixel origin is top-left corner (x=0,y=0)
• Frame size: 1280 px wide × 512 px tall

THREE PHASES — how to identify each:
  "falling"    — droplet is a circular blob in free flight clearly ABOVE the surface line.
                 Centroid is above y ≈ {surface_row} px. Droplet has distinct circular profile.
  "spreading"  — droplet is in contact with the surface; its lower boundary touches the surface line.
                 The droplet is laterally wider than it is tall — an oval or pancake shape.
                 Measure the FULL horizontal width of the dark spreading region in pixels.
  "rebounding" — droplet has separated from the surface and is moving upward.
                 Shape may be elongated vertically (stretched during lift-off).
                 Centroid is at or above the surface line and moving upward.

MEASUREMENT INSTRUCTIONS:
  For falling/rebounding: measure the centroid (cx, cy) and the droplet RADIUS in pixels.
    cx = horizontal centre of the droplet dark blob (0 = left edge of frame)
    cy = vertical centre (0 = top edge of frame)
    radius = half the width of the dark blob (not including the caustic ring)
    Expected cx ≈ 600–700 px (droplet falls near frame centre)
    Expected cy for falling ≈ 150–350 px (above surface)
    Expected radius ≈ {r_px:.0f} px (≈ D₀/2)
  For spreading: measure ONLY spread_width = full horizontal extent of the liquid footprint.
    Expected spread_width ≈ {d0_px:.0f}–{spread_max_px:.0f} px (1× to 3× D₀)

Respond ONLY with JSON:
{{"phase":"falling|spreading|rebounding","cx":number_or_null,"cy":number_or_null,"radius":number_or_null,"spread_width":number_or_null,"confidence":"high|medium|low"}}
"""

def make_task14_prompt(video: str) -> str:
    px_mm = PX_PER_MM_DICT.get(video, 65.625)
    d0_px = 2.0 * px_mm
    r_px = d0_px / 2
    spread_max_px = 3.0 * d0_px
    surface_row = 310  # typical for 02182026 / 03242026 datasets
    return TASK14_PROMPTED_TEMPLATE.format(
        px_per_mm=px_mm,
        mm_per_px=1.0 / px_mm,
        d0_px=d0_px,
        r_px=r_px,
        spread_max_px=spread_max_px,
        surface_row=surface_row,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)


def load_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def image_content(img_path: Path) -> dict:
    b64 = load_image_b64(img_path)
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def call_openrouter(client, model: str, prompt: str, images: list, max_tokens: int = 400) -> str:
    from openai import OpenAI
    content = [image_content(p) for p in images]
    content.append({"type": "text", "text": prompt})
    for attempt in range(3):
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
            print(f"    [retry {attempt+1}/3] {e} — waiting {wait}s")
            time.sleep(wait)
    return ""


def parse_task6(raw: str) -> dict:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m2 = re.search(r"\b([ABCD])\b", raw)
    if m2:
        return {"choice": m2.group(1)}
    return {}


def parse_task14(raw: str) -> dict:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_prompted_eval(model_id: str, client, benchmark: list, delay: float = 0.5):
    slug = model_slug(model_id)
    out_path = RES_DIR / f"prompted_{slug}_results.json"

    # Load existing results to allow resume
    results = []
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"  Resuming — {len(done_ids)} entries already done.")

    # Filter to the 6 target videos only
    target_entries = [
        e for e in benchmark
        if e["video"] in TARGET_VIDEOS
        and e.get("fluid_class") not in (None, "unknown")
        and e["id"] not in done_ids
    ]

    print(f"  Entries to run: {len(target_entries)}")

    for i, entry in enumerate(target_entries):
        task  = entry["task"]
        eid   = entry["id"]
        video = entry["video"]
        imgs  = [BASE / f["path"] for f in entry["frames"] if (BASE / f["path"]).exists()]

        if not imgs:
            print(f"  [{i+1}] SKIP (no frames): {eid}")
            continue

        # Use domain-engineered prompt
        if "task6" in task:
            prompt = TASK6_PROMPTED
        else:
            prompt = make_task14_prompt(video)

        print(f"  [{i+1}/{len(target_entries)}] {task[:6]}  {video:45s}", end="", flush=True)
        raw = call_openrouter(client, model_id, prompt, imgs, max_tokens=500)

        if "task6" in task:
            parsed = parse_task6(raw)
            gt_choice = entry["gt"].get("fluid_choice")
            vlm_choice = parsed.get("choice", "").upper().strip()
            if vlm_choice not in ("A", "B", "C", "D"):
                vlm_choice = None
            result = {
                "id":           eid,
                "task":         task,
                "video":        video,
                "folder":       entry["folder"],
                "fluid_class":  entry["fluid_class"],
                "prompt_type":  "prompted",
                "model":        model_id,
                "gt_choice":    gt_choice,
                "vlm_choice":   vlm_choice,
                "vlm_reasoning": parsed.get("reasoning"),
                "vlm_confidence": parsed.get("confidence"),
                "vlm_raw":      raw,
                "correct":      (vlm_choice == gt_choice) if vlm_choice else False,
            }
            status = f"GT={gt_choice} VLM={vlm_choice} {'✓' if result['correct'] else '✗'}"

        else:
            parsed = parse_task14(raw)
            gt = entry["gt"]
            gt_phase = gt.get("phase")
            vlm_phase = parsed.get("phase")

            def _f(d, k):
                return float(d[k]) if d.get(k) is not None else None

            result = {
                "id":           eid,
                "task":         task,
                "video":        video,
                "folder":       entry["folder"],
                "fluid_class":  entry["fluid_class"],
                "prompt_type":  "prompted",
                "model":        model_id,
                "gt_phase":     gt_phase,
                "gt_cx":        gt.get("cx_px"),
                "gt_cy":        gt.get("cy_px"),
                "gt_radius":    gt.get("radius_px"),
                "gt_spread":    gt.get("spread_px"),
                "vlm_phase":    vlm_phase,
                "vlm_cx":       _f(parsed, "cx"),
                "vlm_cy":       _f(parsed, "cy"),
                "vlm_radius":   _f(parsed, "radius"),
                "vlm_spread":   _f(parsed, "spread_width"),
                "vlm_confidence": parsed.get("confidence"),
                "vlm_raw":      raw,
                "phase_correct": (vlm_phase == gt_phase) if vlm_phase else False,
            }
            for field, gk, vk in [
                ("cx_err", "gt_cx", "vlm_cx"),
                ("cy_err", "gt_cy", "vlm_cy"),
                ("radius_err", "gt_radius", "vlm_radius"),
                ("spread_err", "gt_spread", "vlm_spread"),
            ]:
                g, v = result.get(gk), result.get(vk)
                result[field] = round(abs(v - g), 1) if (g is not None and v is not None) else None
            ph_ok = "✓" if result["phase_correct"] else "✗"
            status = f"phase={vlm_phase}({ph_ok}) r_err={result.get('radius_err')}"

        results.append(result)
        print(f"  {status}")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(delay)

    print(f"  Saved → {out_path}")
    return results


def summarise_results(results: list, model_id: str):
    task6  = [r for r in results if "task6" in r["task"]]
    task14 = [r for r in results if "task1" in r["task"]]

    print(f"\n{'='*60}")
    print(f"PROMPTED RESULTS — {model_id}")

    if task6:
        by_video = {v: None for v in TARGET_VIDEOS}
        for r in task6:
            by_video[r["video"]] = r
        print("\nTask 6 — Fluid classification (6 videos):")
        print(f"  {'Video':<45} GT  Pred  Correct")
        ok = 0
        for v, r in by_video.items():
            if r is None:
                print(f"  {v:<45} —   —     (missing)")
                continue
            correct = r.get("correct", False)
            ok += correct
            mark = "✓" if correct else "✗"
            print(f"  {v:<45} {r['gt_choice']}   {r['vlm_choice'] or '?'}     {mark}")
        print(f"  Accuracy: {ok}/{len([r for r in by_video.values() if r])} = {100*ok/max(1,len(task6)):.0f}%")

    if task14:
        # per-video phase acc and measurement errors
        from collections import defaultdict
        by_video = defaultdict(list)
        for r in task14:
            by_video[r["video"]].append(r)

        print("\nTask 1/4 — Phase + measurement (per video):")
        print(f"  {'Video':<45} PhAcc  cx_MAE  r_MAE  sw_MAE")
        for v in TARGET_VIDEOS:
            vr = by_video.get(v, [])
            if not vr:
                continue
            n = len(vr)
            ph_ok = sum(1 for r in vr if r.get("phase_correct"))
            cx_e  = [r["cx_err"]     for r in vr if r.get("cx_err")     is not None]
            r_e   = [r["radius_err"] for r in vr if r.get("radius_err") is not None]
            sw_e  = [r["spread_err"] for r in vr if r.get("spread_err") is not None]
            def m(x): return f"{np.mean(x):.0f}" if x else "—"
            print(f"  {v:<45} {ph_ok}/{n}   {m(cx_e):>6}  {m(r_e):>5}  {m(sw_e):>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["anthropic/claude-sonnet-4-5", "openai/gpt-4o"],
                        help="Model IDs to evaluate")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)

    with open(BENCHMARK) as f:
        benchmark = json.load(f)

    for model_id in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        results = run_prompted_eval(model_id, client, benchmark, delay=args.delay)
        summarise_results(results, model_id)


if __name__ == "__main__":
    main()
