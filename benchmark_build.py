"""
benchmark_build.py
------------------
Builds the PhysDrop benchmark for two task types:

  Task 1 / Task 4  (phase + measurement, single frame)
      Extracts frames from all 74 videos using the impact frame from
      summary_timeseries_v2.json (02182026 + 03242026) or auto-detection
      (05052026). Ground truth labels come from classical HoughCircles /
      background subtraction — the same pipeline used in vlm_stress_test.py.

  Task 6  (fluid composition classification, 3-frame video-level)
      One entry per video.  Three frames per entry: pre-impact, max-spread,
      post-rebound.  No manual annotation needed — the ground-truth class
      is derived from the filename / experimental design.

      Classes:
          A  pure_water             pure DI water
          B  surfactant_only        surfactant solution, no nanoparticles
          C  CA_with_surfactant     CA nanoparticles + surfactant in droplet
          D  CA_washed              CA nanoparticles, surfactant washed off

Output
------
  benchmark/frames/        PNG files  (<stem>_f<idx:05d>.png)
  benchmark/benchmark.json all entries
  benchmark/stats.json     counts by task / class / folder
"""

import cv2, json, re, sys
import numpy as np
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path("/home/ubuntu/materials")
OUT_DIR    = BASE / "benchmark"
FRAMES_DIR = OUT_DIR / "frames"
TIMESERIES = BASE / "summary_timeseries_v2.json"

FOLDER_MAP = {
    "02182026": BASE / "02182026",
    "03242026": BASE / "03242026_particlesonlypreparedinsurfactant",
    "05052026": BASE / "05052026",
    "05112026": BASE / "new_experiments/05112026",
    "05122026": BASE / "new_experiments/05122026",
}

FPS = 2996.766

# px/mm per folder (new_experiments pending scale calibration; using ~65.6 as fallback)
FOLDER_PX_PER_MM = {
    "02182026": 65.625,
    "03242026": 65.625,
    "05052026": 66.0,
    "05112026": 65.625,
    "05122026": 65.625,
}

# Frames to search for impact (new_experiments videos are much longer)
FOLDER_SEARCH_LIMIT = {
    "02182026": 800,
    "03242026": 800,
    "05052026": 800,
    "05112026": 2000,
    "05122026": 2000,
}

# Default surface row per folder (pixel row of superhydrophobic surface)
FOLDER_DEFAULT_SURFACE_ROW = {
    "02182026": 430,
    "03242026": 470,
    "05052026": 430,
    "05112026": 390,
    "05122026": 330,
}

OUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# ── Fluid class labels ────────────────────────────────────────────────────────

FLUID_CHOICES = {
    "pure_water":         "A",
    "surfactant_only":    "B",
    "CA_with_surfactant": "C",
    "CA_washed":          "D",
}

FLUID_CHOICE_TEXT = {
    "A": "Pure water (DI water, no additives)",
    "B": "Surfactant solution only (no nanoparticles)",
    "C": "CA nanoparticles + surfactant present in droplet",
    "D": "Washed CA nanoparticles (surfactant removed after synthesis)",
}


def get_fluid_class(video: str, folder: str) -> Optional[str]:
    """Return fluid class from filename, or None for scale/calibration videos."""
    n = video.lower()
    stem = n.replace(".mp4", "").strip()

    # Skip calibration
    if stem.startswith("scale"):
        return None

    if folder == "02182026":
        if re.match(r"water\d*$", stem):          return "pure_water"
        if stem == "tx":                           return "surfactant_only"
        if stem.startswith("cainh") or stem.startswith("cainl"):
            return "CA_with_surfactant"
        if stem.startswith("caonly"):              return "CA_washed"

    elif folder == "03242026":
        if "only ca" in stem or stem.startswith("caonly"):
            return "CA_washed"
        if stem == "ca+tr":                        return "CA_washed"
        # pure surfactant controls: "0.001percent cg", "0.028percrnt tx", etc.
        if re.match(r"[\d.]+p", stem) or re.match(r"[\d.]+\s", stem):
            return "surfactant_only"
        if stem.startswith("0.028p"):              return "surfactant_only"

    elif folder == "05052026":
        if stem.startswith("cainh"):               return "CA_with_surfactant"
        # e.g. "0.028tx", "0.08cg", "0.45sds"
        if re.match(r"[\d.]+[a-z]", stem):         return "surfactant_only"

    elif folder in ("05112026", "05122026"):
        # nr50water4 is a scale/ruler calibration recording — no droplet present
        if stem == "nr50water4":                   return None
        if stem.startswith("scale"):               return None
        if stem.startswith("nr50water") or re.match(r"water[\s\d]*$", stem):
            return "pure_water"
        # "ca only 2", "ca only 3"
        if re.match(r"ca only", stem):             return "CA_washed"
        # cain0.028tx*, cain0.08cg*, cain0.45sds*
        if stem.startswith("cain"):                return "CA_with_surfactant"
        # 0.028tx*, 0.45sds*, .08cg* (leading dot — file system artefact)
        if re.match(r"[.\d]+[a-z]", stem):         return "surfactant_only"

    return None


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_timeseries() -> dict:
    with open(TIMESERIES) as f:
        return {r["video"]: r for r in json.load(f)}


# ── Video utilities ───────────────────────────────────────────────────────────

def read_frame(vpath: Path, idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def total_frames(vpath: Path) -> int:
    cap = cv2.VideoCapture(str(vpath))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def detect_impact_frame(vpath: Path, search: int = 800) -> Optional[int]:
    """Largest frame-difference in bottom half = impact."""
    cap  = cv2.VideoCapture(str(vpath))
    nf   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lim  = min(search, nf)
    split = h // 2
    prev, best_i, best_d = None, None, -1
    for i in range(lim):
        ret, fr = cap.read()
        if not ret: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = np.abs(g[split:].astype(np.int16) - prev[split:].astype(np.int16)).mean()
            if d > best_d:
                best_d, best_i = d, i
        prev = g
    cap.release()
    return best_i


def save_png(frame: Optional[np.ndarray], path: Path) -> bool:
    if frame is None or path.exists():
        return path.exists()
    return cv2.imwrite(str(path), frame)


def frame_png_path(stem: str, idx: int) -> Path:
    safe = stem.replace(" ", "_").replace("+", "plus")
    return FRAMES_DIR / f"{safe}_f{idx:05d}.png"


# ── Classical CV ground truth ─────────────────────────────────────────────────

def hough_detect(gray: np.ndarray) -> Optional[tuple]:
    """Return (cx, cy, r) of best circle, or None."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for p2 in [20, 15, 12, 10]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=p2, minRadius=20, maxRadius=95)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            # pick largest circle above the midpoint of frame height
            mid_y = gray.shape[0] // 2
            cands = [x for x in c if x[1] - x[2] < mid_y]
            if not cands:
                cands = list(c)
            best = max(cands, key=lambda x: x[2])
            return float(best[0]), float(best[1]), float(best[2])
    return None


def measure_spread(frame: np.ndarray, bg: np.ndarray, surface_row: int) -> Optional[float]:
    """Background-subtracted contact width at surface level."""
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    r1 = max(0, surface_row - 40)
    r2 = min(mask.shape[0], surface_row + 10)
    band = mask[r1:r2, :]
    cols = np.where(band.max(axis=0) > 0)[0]
    if len(cols) < 5:
        return None
    return float(cols[-1] - cols[0])


def get_background(vpath: Path, impact_frame: int) -> np.ndarray:
    """Median of 5 frames just before impact."""
    frames = []
    for di in range(5, 0, -1):
        fi = max(0, impact_frame - di)
        fr = read_frame(vpath, fi)
        if fr is not None:
            frames.append(fr.astype(np.float32))
    if not frames:
        return np.zeros((512, 1280, 3), dtype=np.float32)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# ── Task 1 / 4: build entries for one video ──────────────────────────────────

MEASURE_PROMPT = (
    "Analyse this high-speed droplet impact frame (1280×512 px).\n\n"
    "The droplet is a dark blob on a bright background. "
    "A horizontal dark line near the bottom of the frame is the superhydrophobic glass surface.\n\n"
    "1. Classify the phase:\n"
    '   "falling"    — droplet in free flight above the surface\n'
    '   "spreading"  — droplet touching surface, spreading outward\n'
    '   "rebounding" — droplet leaving/has left the surface, moving upward\n\n'
    "2. If falling or rebounding: estimate droplet centroid (cx, cy) in pixels "
    "(origin = top-left) and radius in pixels.\n"
    "3. If spreading: estimate the full horizontal contact width in pixels.\n\n"
    "Reply ONLY with JSON:\n"
    '{"phase":"...","cx":null_or_number,"cy":null_or_number,'
    '"radius":null_or_number,"spread_width":null_or_number,"confidence":"high|medium|low"}'
)


def build_task14_video(
    vpath: Path, video: str, folder: str,
    impact: int, liftoff: Optional[int],
    surface_row: int, ts_rec: dict,
    n_falling=5, n_spreading=4, n_rebound=3,
    px_per_mm: float = 65.625,
) -> list[dict]:
    """Return benchmark entries for Task 1/4 for one video."""
    entries = []
    stem = video.replace(".mp4", "")
    fluid_cls = get_fluid_class(video, folder)
    nf = total_frames(vpath)

    phys = {
        "beta_max": ts_rec.get("beta_max"),
        "COR":      ts_rec.get("COR"),
        "D0_mm":    ts_rec.get("D0_mm"),
        "U0_mms":   ts_rec.get("U0_mm_s"),
    }

    bg = get_background(vpath, impact)

    # Sample frame sets
    pre_start = max(0, impact - 18)
    falling_frames = np.linspace(pre_start, impact - 2, n_falling).astype(int).tolist()
    spread_end = liftoff if (liftoff and liftoff < impact + 40) else impact + 20
    spreading_frames = np.linspace(impact + 1, spread_end, n_spreading).astype(int).tolist()
    if liftoff and liftoff + n_rebound * 3 < nf:
        rebound_frames = [liftoff + 3 + i * 4 for i in range(n_rebound)]
    else:
        rebound_frames = [min(impact + 40 + i * 5, nf - 1) for i in range(n_rebound)]

    phase_sets = [
        ("falling",    falling_frames),
        ("spreading",  spreading_frames),
        ("rebounding", rebound_frames),
    ]

    for phase_name, fidxs in phase_sets:
        for fi in fidxs:
            fi = int(fi)
            if fi < 0 or fi >= nf:
                continue
            frame = read_frame(vpath, fi)
            if frame is None:
                continue

            fpath = frame_png_path(stem, fi)
            save_png(frame, fpath)

            # GT labels via classical CV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gt_cx = gt_cy = gt_r = gt_sw = None

            if phase_name in ("falling", "rebounding"):
                res = hough_detect(gray)
                if res:
                    gt_cx, gt_cy, gt_r = res
            elif phase_name == "spreading":
                gt_sw = measure_spread(frame, bg, surface_row)
                # also try hough for the droplet body above surface
                res = hough_detect(gray)
                if res:
                    gt_cx, gt_cy, gt_r = res

            entry = {
                "id":          f"{stem}_f{fi:05d}_t14",
                "task":        "task1_phase_and_measurement",
                "video":       video,
                "folder":      folder,
                "fluid_class": fluid_cls,
                "px_per_mm":   px_per_mm,
                "prompt":      MEASURE_PROMPT,
                "frames": [{
                    "idx":   fi,
                    "path":  str(fpath.relative_to(BASE)),
                    "phase": phase_name,
                }],
                "gt": {
                    "phase":        phase_name,
                    "cx_px":        gt_cx,
                    "cy_px":        gt_cy,
                    "radius_px":    gt_r,
                    "spread_px":    gt_sw,
                    "fluid_class":  fluid_cls,
                    "fluid_choice": FLUID_CHOICES.get(fluid_cls),
                    **phys,
                },
            }
            entries.append(entry)

    return entries


# ── Task 6: fluid classification (3 frames per video) ────────────────────────

TASK6_PROMPT = (
    "You are shown THREE frames from a high-speed shadowgraphy video of a ~2 mm droplet "
    "impacting a superhydrophobic glass surface.\n\n"
    "• Frame 1 — just BEFORE impact (droplet in free fall)\n"
    "• Frame 2 — at MAXIMUM SPREADING (widest contact with surface)\n"
    "• Frame 3 — AFTER REBOUND (droplet leaving or above the surface)\n\n"
    "Study the visual dynamics carefully:\n"
    "  – How far does the droplet spread?\n"
    "  – Does it retract cleanly and bounce, or does it stick?\n"
    "  – Does it fragment or leave a residue?\n"
    "  – What is the droplet shape before and after impact?\n\n"
    "Classify the fluid composition:\n\n"
    "A) Pure water — deionised water, no additives\n"
    "B) Surfactant solution only — water + surfactant (SDS / TX-100 / cocoglycoside), no nanoparticles\n"
    "C) CA nanoparticles + surfactant — cellulose acetate nanoparticles synthesised WITH surfactant, "
    "surfactant still present in the droplet\n"
    "D) Washed CA nanoparticles — CA nanoparticles prepared in surfactant then WASHED (no free surfactant "
    "remains in droplet)\n\n"
    "Reply ONLY with JSON:\n"
    '{"choice":"A|B|C|D","reasoning":"one sentence","confidence":"high|medium|low"}'
)


def build_task6_video(
    vpath: Path, video: str, folder: str,
    impact: int, liftoff: Optional[int],
    ts_rec: dict,
    px_per_mm: float = 65.625,
) -> Optional[dict]:
    """Return one Task 6 entry for this video (3 frames)."""
    fluid_cls = get_fluid_class(video, folder)
    if fluid_cls is None:
        return None

    nf = total_frames(vpath)
    stem = video.replace(".mp4", "")

    # Frame selection
    pre_idx     = max(0, impact - 5)
    spread_idx  = min(impact + 10, nf - 1)
    if liftoff and liftoff + 8 < nf:
        rebound_idx = liftoff + 8
    else:
        rebound_idx = min(impact + 55, nf - 1)

    frame_defs = [
        (pre_idx,    "pre_impact"),
        (spread_idx, "max_spread"),
        (rebound_idx,"post_rebound"),
    ]

    frames_info = []
    for fi, phase_label in frame_defs:
        frame = read_frame(vpath, fi)
        fpath = frame_png_path(stem, fi)
        if save_png(frame, fpath):
            frames_info.append({
                "idx":   fi,
                "path":  str(fpath.relative_to(BASE)),
                "phase": phase_label,
            })

    if len(frames_info) < 2:
        print(f"  [warn] only {len(frames_info)} frames for {video}")
        return None

    phys = {
        "beta_max": ts_rec.get("beta_max"),
        "COR":      ts_rec.get("COR"),
        "D0_mm":    ts_rec.get("D0_mm"),
        "U0_mms":   ts_rec.get("U0_mm_s"),
    }

    return {
        "id":          f"{stem.replace(' ','_')}_task6",
        "task":        "task6_fluid_classification",
        "video":       video,
        "folder":      folder,
        "fluid_class": fluid_cls,
        "px_per_mm":   px_per_mm,
        "prompt":      TASK6_PROMPT,
        "frames":      frames_info,
        "gt": {
            "phase":        None,
            "cx_px":        None,
            "cy_px":        None,
            "radius_px":    None,
            "spread_px":    None,
            "fluid_class":  fluid_cls,
            "fluid_choice": FLUID_CHOICES[fluid_cls],
            **phys,
        },
    }


# ── Surface row lookup (for background subtraction in spreading phase) ────────

SURFACE_ROWS = {
    # 02182026
    "water.mp4":433,"water2.mp4":433,"water3.mp4":433,"water4.mp4":417,
    "water5.mp4":417,"water6.mp4":426,"cainhcg1.mp4":400,"cainhcg2.mp4":433,
    "cainhcg3.mp4":437,"cainhcg4.mp4":433,"cainhcg5.mp4":433,
    "cainhsds1.mp4":433,"cainhsds2.mp4":430,"cainhsds3.mp4":428,
    "cainhtx1.mp4":428,"cainhtx2.mp4":428,"cainhtx3.mp4":402,
    "cainlcg1.mp4":433,"cainlcg2.mp4":433,"cainlcg3.mp4":399,
    "cainlsds1.mp4":427,"cainlsds2.mp4":426,"cainlsds3.mp4":417,
    "cainltx1.mp4":433,"cainltx2.mp4":428,"cainltx3.mp4":422,
    "caonly1.mp4":399,"caonly2.mp4":405,"caonly3.mp4":433,"tx.mp4":417,
    # 03242026
    "0.001percent cg.mp4":404,"0.028p.mp4":404,"0.028percrnt tx.mp4":467,
    "0.45percrnt sds.mp4":454,"ONLY CA SDS ABOVE CMC.mp4":481,
    "ONLY CA SDS ABOVE CMC1.mp4":481,"ONLY CA SDS ABOVE CMC2.mp4":481,
    "ONLY CA cg ABOVE CMC1.mp4":485,"ONLY CA cg ABOVE CMC2.mp4":481,
    "ONLY CA cg ABOVE CMC3.mp4":473,"ONLY CA cg less CMC1.mp4":470,
    "ONLY CA cg less CMC2.mp4":465,"ONLY CA cg less CMC3.mp4":473,
    "ONLY CA sds less CMC1.mp4":471,"ONLY CA sds less CMC2.mp4":470,
    "ONLY CA tx ABOVE CMC1.mp4":482,"ONLY CA tx ABOVE CMC2.mp4":471,
    "ONLY CA tx ABOVE CMC3.mp4":470,"ONLY CA tx ABOVE CMC4.mp4":471,
    "ONLY CA tx less CMC1.mp4":465,"ONLY CA tx less CMC2.mp4":503,
    "ONLY CA tx less CMC3.mp4":505,"ca+TR.mp4":479,
}

DEFAULT_SURFACE_ROW = 430  # last-resort fallback


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading ground-truth timeseries...")
    ts_map = load_timeseries()

    all_entries = []

    for folder_key, folder_path in FOLDER_MAP.items():
        if not folder_path.exists():
            print(f"[skip] missing folder: {folder_path}")
            continue

        mp4s = sorted(folder_path.glob("*.mp4"))
        print(f"\n── {folder_key} ({len(mp4s)} videos) ──")

        folder_px_mm    = FOLDER_PX_PER_MM.get(folder_key, 65.625)
        folder_search   = FOLDER_SEARCH_LIMIT.get(folder_key, 800)
        folder_surf_row = FOLDER_DEFAULT_SURFACE_ROW.get(folder_key, DEFAULT_SURFACE_ROW)

        for vpath in mp4s:
            video = vpath.name
            ts_rec = ts_map.get(video, {})

            # Skip obvious scale/calibration files early
            if get_fluid_class(video, folder_key) is None and \
               not ts_rec.get("impact_frame_ref"):
                print(f"  [skip-scale] {video}")
                continue

            # Get impact frame
            impact = ts_rec.get("impact_frame_ref")
            if impact is None:
                impact = detect_impact_frame(vpath, search=folder_search)
            if impact is None:
                print(f"  [skip] no impact frame: {video}")
                continue
            impact = int(impact)

            liftoff = ts_rec.get("liftoff_frame")
            if liftoff is not None:
                liftoff = int(liftoff)

            surface_row = SURFACE_ROWS.get(video, folder_surf_row)

            # Task 1 / 4
            e14 = build_task14_video(
                vpath, video, folder_key,
                impact, liftoff, surface_row, ts_rec,
                px_per_mm=folder_px_mm,
            )
            all_entries.extend(e14)

            # Task 6
            e6 = build_task6_video(
                vpath, video, folder_key,
                impact, liftoff, ts_rec,
                px_per_mm=folder_px_mm,
            )
            if e6:
                all_entries.append(e6)

            label14 = len(e14)
            label6  = "yes" if e6 else "no"
            cls     = get_fluid_class(video, folder_key) or "—"
            print(f"  {video:45s}  impact={impact:5d}  t14={label14:2d}  t6={label6}  class={cls}")

    # ── Save ──────────────────────────────────────────────────────────────────
    bench_path = OUT_DIR / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    # ── Stats ─────────────────────────────────────────────────────────────────
    task_counts   = {}
    class_counts  = {}
    folder_counts = {}
    t6_choices    = {"A":0,"B":0,"C":0,"D":0}

    for e in all_entries:
        t = e["task"];   task_counts[t]   = task_counts.get(t, 0) + 1
        c = e.get("fluid_class") or "unknown"
        class_counts[c] = class_counts.get(c, 0) + 1
        fld = e.get("folder") or "?"
        folder_counts[fld] = folder_counts.get(fld, 0) + 1
        if e["task"] == "task6_fluid_classification":
            ch = e["gt"].get("fluid_choice","?")
            t6_choices[ch] = t6_choices.get(ch, 0) + 1

    stats = {
        "total": len(all_entries),
        "by_task": task_counts,
        "by_fluid_class": class_counts,
        "by_folder": folder_counts,
        "task6_choice_distribution": t6_choices,
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total entries  : {stats['total']}")
    print(f"By task        : {stats['by_task']}")
    print(f"By fluid class : {stats['by_fluid_class']}")
    print(f"By folder      : {stats['by_folder']}")
    print(f"Task6 choices  : {stats['task6_choice_distribution']}")
    print(f"\nBenchmark → {bench_path}")
    print(f"Frames    → {FRAMES_DIR}")


if __name__ == "__main__":
    main()
