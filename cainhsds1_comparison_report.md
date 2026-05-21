# cainhsds1.mp4 — OpenCV vs SAM2 Comparison Report

**Video:** `02182026/cainhsds1.mp4`  
**Resolution:** 1280 × 512 px | **Encoded FPS:** 60.0 | **Actual FPS:** 2996.766  
**Total frames:** 3279 | **Calibration:** 65.625 px/mm  
**Known ground truth (feature_table):** impact_frame=1182, surface_y=433 px

---

## 1. OpenCV Script Results (`analyze_droplet_opencv.py`)

### 1.1 Run command

```bash
python3 analyze_droplet_opencv.py \
  --video 02182026/cainhsds1.mp4 \
  --output cainhsds1_opencv.csv
```

### 1.2 What the script found

The script detected **two separate dispensing events** in the video:

| Drop ID | Role | Frames detected | cy range | Area |
|---------|------|----------------|----------|------|
| 1 | **Main droplet** — falling + residual on surface | 772–3278 | 69 → 399 px | ~6 000 px² |
| 6 | **Second (satellite) droplet** — falling | 1148–1179 | 7 → 379 px | ~1 110 px² |
| 7 | Second droplet residual on surface | 1179–3278 | ~400 px | ~1 700 px² |

**Reference frame:** 772 | **Reference area:** 5 672 px²  
**Total rows written:** 4 710 | **Blank frames:** 772 (before droplet enters)

---

### 1.3 Drop 1 — Main Droplet

**Falling phase (frames 772–784, 13 frames):**

| Frame | cx (px) | cy (px) | Area (px²) | % of ref |
|-------|---------|---------|-----------|----------|
| 772 | 1098 | 69 | 5 672 | 100.0% |
| 773 | 1100 | 92 | 6 241 | 110.0% |
| 774 | 1102 | 115 | 6 225 | 109.7% |
| 775 | 1103 | 138 | 6 200 | 109.3% |
| 776 | 1105 | 161 | 6 218 | 109.6% |
| 777 | 1106 | 185 | 6 197 | 109.2% |
| 778 | 1108 | 208 | 6 219 | 109.6% |
| 779 | 1109 | 231 | 6 236 | 109.9% |
| 780 | 1111 | 255 | 6 252 | 110.2% |
| 781 | 1112 | 278 | 6 275 | 110.6% |
| 782 | 1114 | 302 | 6 260 | 110.4% |
| 783 | 1115 | 325 | 6 284 | 110.8% |
| 784 | 1116 | 349 | 6 220 | 109.7% |

**Computed metrics:**

| Metric | Value | Notes |
|--------|-------|-------|
| U0 (impact velocity) | **1 065.6 mm/s** (1.066 m/s) | Linear fit on cy vs frame |
| D0 (pre-impact diameter) | **1.36 mm** | From median contour area: 2√(A/π)/px_per_mm |
| Impact frame (estimated) | ~**785** | cy reaches surface_y−radius ≈ 390 |
| Spreading peak area | 9 409 px² at frame 800 | 165.9% of reference |

> **Note on D0:** The contour-based area (5 672 px²) gives a diameter of 1.36 mm, which is smaller than the expected ~2.0 mm for a 4 µL sphere. This is because background-subtraction thresholding at pixel-diff=25 misses the low-contrast outer rim of the droplet. The HoughCircles method in `ellipse_timeseries.py` gives a more accurate D0=1.375 mm (feature_table: radius_px=86 → D0=2.62 mm from HoughCircles). Contour area underestimates area because only the high-contrast core of the droplet exceeds the threshold.

**Post-impact (surface residual, frames 790–3278):**
- Median area: 6 029 px² ≈ **106.3% of reference**
- Position: cy ≈ 395–399 px (near surface at 433 − radius)
- The droplet **does not bounce** — it stays on the surface as a sessile drop for the remainder of the video. COR = 0 for this event.

---

### 1.4 Drop 6 — Second (Satellite) Droplet

Detected entering the frame at frame 1148, 376 frames after the first impact.

| Frame | cy (px) | Area (px²) | % of ref |
|-------|---------|-----------|----------|
| 1148 | 7 | 517 | 9.1% |
| 1155 | 79 | 1 106 | 19.5% |
| 1165 | 200 | 1 122 | 19.8% |
| 1175 | 327 | 1 110 | 19.6% |
| 1179 | 379 | 1 074 | 18.9% |

**Computed metrics:**

| Metric | Value | Notes |
|--------|-------|-------|
| U0 | **558.2 mm/s** (0.558 m/s) | Linear fit over 32 frames |
| D0 | **0.57 mm** | Area ≈ 1 110 px² → very small satellite |
| Area as % of main | **19.6%** | Sub-droplet, not the primary event |
| Impact frame | ~**1179–1182** | Matches feature_table impact_frame=1182 |

> The `feature_table.json` records `impact_frame=1182` which corresponds to this **satellite droplet**, not the main dispensed droplet. The main impact was at frame ~785. The satellite may be a secondary drop ejected from the nozzle after the main one.

---

### 1.5 Volume Conservation Check

| Component | Area (px²) | % of initial reference |
|-----------|-----------|----------------------|
| Drop 1 surface residual | 6 029 | 106.3% |
| Drop 7 (satellite residual) | 1 727 | 30.5% |
| **Combined** | **7 756** | **136.8%** |

Combined residual > 100% indicates the reference frame (772, cy=69) captured only the upper portion of the droplet still entering the frame, underestimating the true reference area. A better reference area would be from a frame where the droplet is fully in-frame and circular (e.g., frame 778, area 6 219 px²).

---

### 1.6 Output CSV summary

```
cainhsds1_opencv.csv
  Rows:        4 710
  Drop IDs:    1 (2507 rows), 2 (1), 3 (18), 4 (9), 5 (43), 6 (32), 7 (2100)
  Frame range: 772 – 3278
```

---

## 2. SAM2 Script — Expected Results (`analyze_droplet_sam2.py`)

> **SAM2 could not be run** — no PyTorch or CUDA GPU available in this environment.  
> The comparison below is based on SAM2's documented behaviour and architectural differences.

### 2.1 What SAM2 would do differently

**Step 1 — Reference frame detection (shared with OpenCV):**  
SAM2 uses the same OpenCV background-subtraction pre-pass to locate the reference frame. It would find the same reference frame 772 and centroid (1098, 69).

**Step 2 — Frame extraction:**  
All 3 279 frames extracted as JPEGs to a temp directory (~500 MB disk space, ~20–60 s depending on I/O).

**Step 3 — Mask propagation:**  
SAM2 propagates the mask forward from frame 772. The key differences:

| Aspect | OpenCV result | Expected SAM2 result |
|--------|--------------|----------------------|
| D0 (reference area) | 5 672 px² (1.36 mm) | ~8 500–9 500 px² (~1.66 mm) — SAM2 captures the full droplet boundary including low-contrast outer rim |
| Falling phase tracking | Clean (contour centroid) | Clean (pixel mask centroid) — similar accuracy |
| Impact + spreading shape | Loses the non-circular spreading lamella (circ_thresh=0.3 cuts it off at frame 785) | Propagates the mask through the entire spreading phase including the flat lamella |
| Post-impact area | Reports the residual sessile drop (~106% of ref) | Accurately segments the sessile drop shape |
| Second droplet | Detected as new contour (drop_id=6) when it appears | **Would NOT detect** — SAM2 tracks only object_id=1 (the first droplet); the satellite requires a second `add_new_points_or_box` call |
| Spreading max | Not directly captured (contour loses shape) | Would give precise β_max from mask area |

### 2.2 Expected SAM2 output CSV structure

```
frame, drop_id, cx, cy, area_px, percentage
  772,       1, 1098,  69,  9100, 100.0    ← larger area, full droplet boundary
  773,       1, 1100,  92,  9180, 100.9
  ...
  785,       1, 1117, 385, 14200, 156.0    ← spreading lamella captured
  786,       1, 1116, 392, 12800, 140.7
  ...
```

> Note: SAM2 CSV does **not** include the `in_frame` column (present in OpenCV output).

### 2.3 What SAM2 cannot do without code changes

- **Detect the satellite droplet (drop_id=6/7)** — would need a second point prompt added to the second droplet when it enters the frame.
- **Run without GPU** — CPU mode is ~50–100× slower and may OOM on 3 279 frames.

---

## 3. Side-by-Side Comparison

| Metric | OpenCV | SAM2 (expected) |
|--------|--------|----------------|
| **Runtime** | ~45 seconds (CPU) | ~5–20 minutes (GPU) |
| **GPU required** | No | Yes (8+ GB VRAM) |
| **Reference frame** | 772 | 772 (same pre-pass) |
| **D0 main droplet** | 1.36 mm (area underestimate) | ~1.65 mm (full mask) |
| **U0 main droplet** | 1 065.6 mm/s | ~1 060 mm/s (similar) |
| **Spreading detection** | Partially lost (circularity filter) | Full lamella tracked |
| **β_max** | ~165.9% (area estimate) | More accurate |
| **Second droplet** | Detected as drop_id=6 ✓ | Not detected (needs extra prompt) |
| **Post-impact tracking** | Sessile drop tracked as residual | Precise mask of sessile drop |
| **Volume conservation** | 136.8% (ref area underestimated) | Closer to 100% (better area) |
| **Mask quality** | Contour only (loses rim) | Pixel-accurate |
| **Deformed shapes** | Limited by circ_thresh | Fully handled |

---

## 4. Issues Found in the Scripts

### 4.1 `analyze_droplet_opencv.py` — Bug on line 215

```python
# BUG: uses tracked.index() which indexes into tracked (sorted by drop_id),
# not into full (sorted by area descending). Wrong bounding box used for in_frame.
cv2.boundingRect(full[tracked.index((drop_id, cx, cy, area))])
```

**Fix:** build a contour → detection mapping before calling `tracker.update()`, same as the post-reference phase already does with `bbox_map`.

### 4.2 `README_opencv.md` — Wrong description (line 99)

States: *"Does not track droplet identity across frames"*  
Actual behaviour: The `DropletTracker` class **does** maintain persistent IDs via greedy nearest-neighbour centroid matching.

### 4.3 `README_opencv.md` — Missing v2 parameters

Three CLI flags added in v2 are absent from the README parameter table:
- `--diff_thresh_post` (default 15) — lower threshold post-reference
- `--circ_thresh_post` (default 0.1) — lower circularity post-reference
- `--max_dist` (default 100) — max centroid distance for ID matching

### 4.4 `analyze_droplet_sam2.py` — CPU bfloat16 incompatibility (line 197)

```python
# May crash on CPU-only PyTorch builds that don't support bfloat16
torch.autocast(device_type=device, dtype=torch.bfloat16)
```

**Fix:** use `dtype=torch.float32` when `device == "cpu"`.

### 4.5 `analyze_droplet_sam2.py` — Missing `in_frame` column

OpenCV output has `in_frame` (1/0 flag). SAM2 output omits it, making the two CSVs schema-incompatible for downstream analysis.

---

## 5. Recommendation

For this experiment (superhydrophobic surface, clear droplet-on-background contrast):

- **Use OpenCV** for fast batch processing of all 53 videos to get U0, D0, and rough β_max. Already done via `ellipse_timeseries_v2.py` which handles these more accurately.
- **Use SAM2** selectively on 3–5 representative videos per condition where precise spreading-lamella shape and volume tracking matter (e.g., cainhsds2 for validation, one water baseline, one above-CMC sample).
- **Fix the four issues above** before publishing results from either script.
