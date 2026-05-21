# Droplet Impact Analysis — Full Documentation

**Project:** Superhydrophobic surface wetting experiments with cellulose acetate nanoparticles  
**Date compiled:** 2026-04-12  
**Drop volume:** 4 µL | **Drop height:** 6.6 cm | **Surface:** Superhydrophobic coating on glass slide

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Camera & Video Properties](#2-camera--video-properties)
3. [Scale Calibration](#3-scale-calibration)
4. [Parameters to Extract](#4-parameters-to-extract)
5. [Classical Computer Vision Pipeline](#5-classical-computer-vision-pipeline)
6. [AI & Deep Learning Methods](#6-ai--deep-learning-methods)
7. [Key Dimensionless Numbers](#7-key-dimensionless-numbers)
8. [Recent Literature](#8-recent-literature)
9. [Recommended Workflow](#9-recommended-workflow)
10. [Next Steps](#10-next-steps)

---

## 1. Dataset Overview

### Folder: `02182026/`

CA nanoparticles **with surfactant present** in the droplet, dropped on superhydrophobic coating.

| Sample code | Description |
|---|---|
| `cainhcg1–5` | CA + **high** CG (cocoglycoside, 2× CMC = 0.01 wt%) |
| `cainhsds1–3` | CA + **high** SDS (2× CMC = 0.45 wt%) |
| `cainhtx1–3` | CA + **high** TX-100 (2× CMC = 0.028 wt%) |
| `cainlcg1–3` | CA + **low** CG (0.5× CMC = 0.003 wt%) |
| `cainlsds1–3` | CA + **low** SDS (0.5× CMC = 0.112 wt%) |
| `cainltx1–3` | CA + **low** TX-100 (0.5× CMC = 0.007 wt%) |
| `caonly1–3` | CA particles in DI water only (no surfactant) |
| `water1–6` | Pure DI water (baseline control) |
| `tx.mp4` | TX-100 surfactant solution only |
| `scale.mp4` | Horizontal ruler for pixel calibration |
| `scale v.mp4` | Vertical ruler for pixel calibration |
| `Spreading/` | Additional spreading videos |

**CMC reference values:**
- SDS: 0.225 wt%
- TX-100: 0.014 wt%
- Cocoglycoside (CG): 0.005 wt%

---

### Folder: `03242026_particlesonlypreparedinsurfactant/`

CA nanoparticles where **surfactant was removed after synthesis** (washed by centrifugation × 3). Only particle morphology effect is isolated — no residual surfactant in droplet.

| Sample code | Description |
|---|---|
| `ONLY CA SDS ABOVE CMC 1–3` | Particles made in SDS > CMC, surfactant removed |
| `ONLY CA sds less CMC 1–3` | Particles made in SDS < CMC, surfactant removed |
| `ONLY CA tx ABOVE CMC 1–4` | Particles made in TX > CMC, surfactant removed |
| `ONLY CA tx less CMC 1–3` | Particles made in TX < CMC, surfactant removed |
| `ONLY CA cg ABOVE CMC 1–3` | Particles made in CG > CMC, surfactant removed |
| `ONLY CA cg less CMC 1–3` | Particles made in CG < CMC, surfactant removed |
| `0.001percent cg.mp4` | Pure CG surfactant solution (no particles) |
| `0.028percrnt tx.mp4` | Pure TX surfactant solution (no particles) |
| `0.45percrnt sds.mp4` | Pure SDS surfactant solution (no particles) |
| `ca+TR.mp4` | CA + tracer particles |

> **Same magnification as 02182026 folder** — scale calibration from `scale.mp4` and `scale v.mp4` applies to both folders.

---

## 2. Camera & Video Properties

| Property | Value |
|---|---|
| Resolution | 1280 × 512 px |
| Encoded/playback FPS | ~60 fps |
| **Actual capture FPS** | **~4000 fps** (estimated from droplet physics — see below) |
| Inter-frame time | ~0.25 ms |
| Format | MP4 |

### How the actual FPS was estimated

The MP4 files report ~60 fps, which is the **playback encoding rate** used by Photron/Phantom-type high-speed cameras. The actual capture rate was estimated by:

1. Detecting droplet centroid position (y in pixels) in frames 423–439 of `water.mp4` using HoughCircles
2. Fitting a parabola: $y(f) = a f^2 + b f + c$, giving $a = 0.0213$ px/frame²
3. The coefficient $a = \frac{1}{2} \cdot \frac{g \cdot \text{px/m}}{\text{fps}^2}$
4. With $g = 9.81\ \text{m/s}^2$ and droplet radius ≈ 69 px ≈ 1 mm (4 µL sphere), giving 69,000 px/m:

$$\text{fps}_{actual} = \sqrt{\frac{g \cdot \text{px/m}}{2a}} = \sqrt{\frac{9.81 \times 69000}{0.0426}} \approx 4000\ \text{fps}$$

> **Action required:** Confirm the exact capture FPS from the camera's export metadata or operator notes. Common values at 1280×512 are **2000, 3200, or 4000 fps**. All velocity calculations depend on this number.

---

## 3. Scale Calibration

### 3.1 Horizontal calibration — `scale.mp4` ✅ MEASURED

**Method:** Column-wise intensity profiling + peak detection on the ruler image.

- Extracted frame 0 from `scale.mp4`
- Applied column mean across rows 60–420 (the tick region)
- Used `scipy.signal.find_peaks` on the inverted intensity profile to locate dark tick positions
- **17 ticks detected** at columns: 141, 210, 274, 338, 404, 471, 536, 602, 668, 735, 800, 867, 932, 997, 1063, 1127, 1191
- Span: 1191 − 141 = **1050 px over 16 intervals = 16 mm**

| Metric | Value |
|---|---|
| **px_per_mm (horizontal)** | **65.625 px/mm** |
| mm_per_px (horizontal) | 0.01524 mm/px |
| Tick spacing std | ±1.36 px (2.1% variation — excellent) |

All 17 red annotation lines were visually confirmed to align precisely with ruler ticks.

**Validation against droplet size:**  
4 µL sphere diameter = $2 \times \left(\frac{3 \times 4 \times 10^{-9}}{4\pi}\right)^{1/3} = 2.006\ \text{mm}$  
HoughCircles on pre-impact frames gave radius ≈ 69 px → diameter = 138 px → 138 / 65.625 = **2.10 mm** ✓ (within 5% of theoretical)

---

### 3.2 Vertical calibration — `scale v.mp4` ✅ RESOLVED

**Ruler type (from `samples.txt`):** Imperial ruler — **1 graduation = 1/16 inch**. The frame shows the **12-inch mark**.

**Detected tick rows:** 43, 147, 253, 356 (3 consistent intervals)  
**Mean spacing:** 104.3 px/tick  
**1/16 inch in mm:** 25.4 / 16 = **1.5875 mm per graduation**

**Derived vertical calibration:**
$$\text{px/mm} = \frac{104.3\ \text{px}}{1.5875\ \text{mm}} = 65.7\ \text{px/mm}$$

| Metric | Value |
|---|---|
| **px_per_mm (vertical)** | **65.7 px/mm** |
| mm_per_px (vertical) | 0.01522 mm/px |
| Tick spacing (imperial) | 1/16 inch = 1.5875 mm |

**Validation:** Predicted droplet vertical diameter = 138 px / 65.7 = **2.10 mm** vs. theoretical 2.01 mm ✓

> **Horizontal and vertical calibrations agree to within 0.1%** (65.625 vs 65.7 px/mm) — confirming the camera distance was held constant between the two scale recordings.

---

### 3.3 Summary — calibration values to use

```python
PX_PER_MM_HORIZONTAL = 65.625   # MEASURED from horizontal mm ruler (17 ticks / 16 mm)
PX_PER_MM_VERTICAL   = 65.700   # MEASURED from vertical imperial ruler (1/16 inch ticks)
PX_PER_MM            = 65.625   # Use this single value for all measurements (< 0.1% difference)
MM_PER_PX            = 1 / 65.625  # = 0.015238 mm/px
```

---

## 3b. Surface Y-Row Detection ✅ MEASURED

The superhydrophobic glass slide appears as a strong horizontal edge in each video. Because the slide was repositioned between recording sessions, the surface row **varies per video** and must be detected individually.

### Method

For each video, 5 frames from the first 25% (before any impact) are averaged. A horizontal Sobel filter detects horizontal edges. The strongest persistent edge in rows 350–511 is taken as the surface.

```python
from scipy.signal import find_peaks

def detect_surface_row(video_path, search_rows=(350, 511), n_frames=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_edges = np.zeros(512)
    count = 0
    sample_frames = [int(total * f) for f in [0.05, 0.10, 0.15, 0.20, 0.25]]
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        center = np.abs(sobel[:, 200:1080])
        all_edges += center.mean(axis=1)
        count += 1
    cap.release()
    avg_edges = all_edges / count
    search = avg_edges[search_rows[0]:search_rows[1]]
    peaks, _ = find_peaks(search, height=5, distance=15)
    best = peaks[np.argmax(search[peaks])]
    return best + search_rows[0]
```

### Measured surface rows — `02182026/`

| Video | Surface row (px) |
|---|---|
| water.mp4 | 433 |
| water2.mp4 | 433 |
| water3.mp4 | 433 |
| water4.mp4 | 417 |
| water5.mp4 | 417 |
| water6.mp4 | 426 |
| cainhcg1.mp4 | 400 |
| cainhcg2.mp4 | 433 |
| cainhcg3.mp4 | 437 |
| cainhcg4.mp4 | 433 |
| cainhcg5.mp4 | 433 |
| cainhsds1.mp4 | 433 |
| cainhsds2.mp4 | 430 |
| cainhsds3.mp4 | 428 |
| cainhtx1.mp4 | 428 |
| cainhtx2.mp4 | 428 |
| cainhtx3.mp4 | 402 |
| cainlcg1.mp4 | 433 |
| cainlcg2.mp4 | 433 |
| cainlcg3.mp4 | 399 |
| cainlsds1.mp4 | 427 |
| cainlsds2.mp4 | 426 |
| cainlsds3.mp4 | 417 |
| cainltx1.mp4 | 433 |
| cainltx2.mp4 | 428 |
| cainltx3.mp4 | 422 |
| caonly1.mp4 | 399 |
| caonly2.mp4 | 405 |
| caonly3.mp4 | 433 |
| tx.mp4 | 417 |

### Measured surface rows — `03242026_particlesonlypreparedinsurfactant/`

| Video | Surface row (px) |
|---|---|
| 0.001percent cg.mp4 | 404 |
| 0.028p.mp4 | 404 |
| 0.028percrnt tx.mp4 | 467 |
| 0.45percrnt sds.mp4 | 454 |
| ONLY CA SDS ABOVE CMC.mp4 | 481 |
| ONLY CA SDS ABOVE CMC1.mp4 | 481 |
| ONLY CA SDS ABOVE CMC2.mp4 | 481 |
| ONLY CA cg ABOVE CMC1.mp4 | 485 |
| ONLY CA cg ABOVE CMC2.mp4 | 481 |
| ONLY CA cg ABOVE CMC3.mp4 | 473 |
| ONLY CA cg less CMC1.mp4 | 470 |
| ONLY CA cg less CMC2.mp4 | 465 |
| ONLY CA cg less CMC3.mp4 | 473 |
| ONLY CA sds less CMC1.mp4 | 471 |
| ONLY CA sds less CMC2.mp4 | 470 |
| ONLY CA tx ABOVE CMC1.mp4 | 482 |
| ONLY CA tx ABOVE CMC2.mp4 | 471 |
| ONLY CA tx ABOVE CMC3.mp4 | 470 |
| ONLY CA tx ABOVE CMC4.mp4 | 471 |
| ONLY CA tx less CMC1.mp4 | 465 |
| ONLY CA tx less CMC2.mp4 | 503 |
| ONLY CA tx less CMC3.mp4 | 505 |
| ca+TR.mp4 | 479 |

> **Note:** The `03242026` folder has surface rows consistently ~40–70 px lower (larger row numbers) than `02182026`, confirming the slide was repositioned lower in the frame between the two recording sessions.

---

## 4. Parameters to Extract

### 4.1 Initial Velocity (pre-impact)

**Definition:** Velocity of the droplet just before it contacts the surface.

**Expected value from free-fall:**
$$v_0 = \sqrt{2gh} = \sqrt{2 \times 9.81 \times 0.066} = 1.137\ \text{m/s}$$

The actual velocity may differ slightly due to pipette ejection thrust (can be positive or negative deviation). The field of view is limited — the droplet typically enters the frame near the top and impacts ~10–15 frames later.

**Extraction method:**
1. Find impact frame $f_{impact}$ (large frame-to-frame pixel difference at the bottom of frame)
2. Detect droplet centroid $(x_c, y_c)$ in the 5–10 frames **before** impact
3. Fit a linear regression to $y_c$ vs. frame number:
$$v_0 = \frac{\Delta y_{px}}{\Delta f} \times \frac{\text{fps}_{actual}}{\text{px/m}}$$

---

### 4.2 Post-impact Rebound Velocity

**Definition:** Velocity of the droplet (or main fragment) immediately after it lifts off the surface.

**Extraction method:**
1. Identify the first frame where the droplet detaches from the surface (liftoff frame $f_{liftoff}$)
2. Track centroid in the 3–8 frames after liftoff (upward motion = decreasing $y$)
3. Fit linear regression to $y_c$ vs. frame (same formula as above, but in upward direction)

**Derived metric — Coefficient of Restitution:**
$$e = \frac{v_{rebound}}{v_{impact}}$$

$e$ ranges from 0 (complete stick) to ~0.9 (near-perfect bounce). This will be the primary comparative metric between your samples.

---

### 4.3 Droplet Diameter Before and After Impact

**Before impact ($D_0$):**
- Detect the droplet as a circle in pre-impact frames using HoughCircles or contour fitting
- Measure radius $r_{px}$, convert: $D_0 = 2 \times r_{px} / \text{px\_per\_mm}$ [mm]
- Expected: ~2.0 mm for a 4 µL sphere

**After impact — fragmentation:**
- Some samples will show fragmentation (part of droplet left on surface, part bounces)
- Detect all contours in post-bounce frames
- For each fragment: $D_{frag} = 2\sqrt{A_{px}/\pi} / \text{px\_per\_mm}$ where $A_{px}$ is contour area
- Report: diameter of bouncing fragment, diameter of residual on surface (if any)

**Volume conservation check:**
$$V_{bounce} + V_{residual} \approx V_0 = 4\ \mu\text{L}$$

---

### 4.4 Maximum Spreading Diameter

**Definition:** The largest horizontal extent of the droplet while it is in contact with the surface ("pancake" phase).

**Extraction method:**
1. Scan all frames from first contact to liftoff
2. In each frame, find the horizontal width of the droplet/contact region at the surface level
3. $D_{max}$ = maximum width found across all contact frames, in mm

**Key derived metric — Spreading Factor:**
$$\beta_{max} = \frac{D_{max}}{D_0}$$

Typical values on superhydrophobic surfaces: 1.5–4. Higher $\beta_{max}$ = more spreading, less bouncing.

---

## 5. Classical Computer Vision Pipeline

### Complete Python workflow (OpenCV + scipy)

```python
import cv2
import numpy as np
from scipy.optimize import curve_fit

# ── STEP 1: Scale calibration ──────────────────────────────────────────────
def get_scale(scale_video_path, known_tick_spacing_mm):
    cap = cv2.VideoCapture(scale_video_path)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find vertical edges (ruler ticks) via gradient
    grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Peak detection on column sum → tick positions
    # Measure pixel distance between consecutive ticks
    px_per_mm = ...  # fill from tick detection
    return px_per_mm

# ── STEP 2: Find impact frame ──────────────────────────────────────────────
def find_impact_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    prev = None
    diffs = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            # Focus on bottom quarter of frame (where surface is)
            roi = gray[350:, :]
            roi_prev = prev[350:, :]
            diffs.append((i, np.abs(roi.astype(int) - roi_prev.astype(int)).mean()))
        prev = gray
        i += 1
    cap.release()
    diffs.sort(key=lambda x: x[1], reverse=True)
    return diffs[0][0]  # frame with largest bottom-region change

# ── STEP 3: Track droplet centroid ────────────────────────────────────────
def track_droplet(video_path, start_frame, end_frame, min_r=15, max_r=80):
    cap = cv2.VideoCapture(video_path)
    positions = []
    for fi in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=50, param2=20, minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            c = np.round(circles[0]).astype(int)
            # Take uppermost circle (smallest y) as droplet
            best = sorted(c, key=lambda x: x[1])[0]
            positions.append((fi, best[0], best[1], best[2]))  # frame, x, y, r
    cap.release()
    return positions  # list of (frame, cx, cy, radius_px)

# ── STEP 4: Compute velocity from centroid positions ──────────────────────
def compute_velocity(positions, fps_actual, px_per_m, direction='down'):
    frames = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[2] for p in positions], dtype=float)
    # Linear fit: y = m*f + b → velocity in px/frame
    coeffs = np.polyfit(frames, ys, 1)
    v_px_per_frame = coeffs[0]
    if direction == 'up':
        v_px_per_frame = -v_px_per_frame
    v_m_per_s = v_px_per_frame * fps_actual / px_per_m
    return v_m_per_s

# ── STEP 5: Measure pre-impact diameter ───────────────────────────────────
def measure_diameter(positions, px_per_mm):
    radii = [p[3] for p in positions]
    r_px = np.median(radii)
    D0_mm = 2 * r_px / px_per_mm
    return D0_mm

# ── STEP 6: Measure maximum spreading diameter ───────────────────────────
def measure_max_spreading(video_path, impact_frame, end_frame, px_per_mm, surface_y):
    cap = cv2.VideoCapture(video_path)
    max_width_px = 0
    for fi in range(impact_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        # Look at a thin horizontal band at the surface
        band = thresh[surface_y - 10:surface_y + 5, :]
        cols = np.where(band.max(axis=0) > 0)[0]
        if len(cols) > 0:
            width = cols[-1] - cols[0]
            max_width_px = max(max_width_px, width)
    cap.release()
    D_max_mm = max_width_px / px_per_mm
    return D_max_mm

# ── STEP 7: Full pipeline for one video ──────────────────────────────────
def analyze_video(video_path, fps_actual, px_per_mm):
    px_per_m = px_per_mm * 1000
    impact_frame = find_impact_frame(video_path)

    # Pre-impact: track 10 frames before impact
    pre = track_droplet(video_path, impact_frame - 12, impact_frame - 2)
    v_initial = compute_velocity(pre, fps_actual, px_per_m, direction='down')
    D0_mm = measure_diameter(pre, px_per_mm)

    # Post-bounce: track 8 frames after liftoff (approx impact + 20)
    liftoff_frame = impact_frame + 18  # adjust per video
    post = track_droplet(video_path, liftoff_frame, liftoff_frame + 8)
    v_rebound = compute_velocity(post, fps_actual, px_per_m, direction='up')

    # Max spreading
    surface_y = 420  # pixel row of surface — adjust per video
    D_max_mm = measure_max_spreading(video_path, impact_frame, impact_frame + 20,
                                     px_per_mm, surface_y)

    # Derived quantities (assume water: rho=998, sigma=0.072, mu=0.001)
    rho, sigma, mu = 998, 0.072, 0.001
    D0_m = D0_mm / 1000
    We = rho * v_initial**2 * D0_m / sigma
    Re = rho * v_initial * D0_m / mu
    e = v_rebound / v_initial
    beta_max = D_max_mm / D0_mm

    return {
        'v_initial_m_s': round(v_initial, 4),
        'v_rebound_m_s': round(v_rebound, 4),
        'D0_mm': round(D0_mm, 3),
        'D_max_mm': round(D_max_mm, 3),
        'We': round(We, 2),
        'Re': round(Re, 1),
        'e': round(e, 4),
        'beta_max': round(beta_max, 3),
    }
```

> **Note on surface_y and liftoff_frame:** These will vary between videos. The recommended approach is to detect the surface line automatically (horizontal edge at bottom of frame) and detect liftoff by checking when the droplet contour detaches from the surface row.

---

## 6. AI & Deep Learning Methods

This section covers state-of-the-art machine learning and deep learning approaches directly applicable to your parameter extraction task.

---

### 6.1 Object Detection & Tracking — YOLO + DeepSORT

**Most directly applicable for automated droplet tracking across all your videos.**

#### DropTrack (YOLOv5 + DeepSORT)

- **Paper:** Durve et al., *"DropTrack — Automatic droplet tracking with YOLOv5 and DeepSORT for microfluidic applications"*, Physics of Fluids, Vol. 34, 082003 (2022)
- **Architecture:** YOLOv5 detector + DeepSORT multi-object tracker
- **Key finding:** Best accuracy achieved with 40% real + 60% synthetically generated training images
- **Capability:** Tracks droplets in dense emulsions; extracts full trajectories, diameter, velocity
- **Relevance to your work:** Directly applicable. Can track the droplet through pre-impact, spreading, and rebound phases automatically

**Benchmarking study:**
- **Paper:** *"Benchmarking YOLOv5 and YOLOv7 models with DeepSORT for droplet tracking"*, European Physical Journal E (2023)
- **Finding:** YOLOv7 is ~10% faster than YOLOv5; real-time tracking needs a mid-range GPU

**Practical steps to apply DropTrack to your videos:**
1. Annotate ~200–300 frames from your videos (droplet bounding boxes) using LabelImg or CVAT
2. Augment with synthetic frames (Gaussian blur, brightness variation)
3. Fine-tune YOLOv5/YOLOv8 on your dataset
4. Run DeepSORT to get continuous trajectories → extract position vs. frame → differentiate for velocity

---

### 6.2 Instance Segmentation — Mask R-CNN & SAM

Gives pixel-accurate droplet boundaries (better than bounding boxes for diameter/spreading measurement).

#### Mask R-CNN for Droplet Detection

- **Paper:** *"Mask R-CNN based droplet detection in liquid–liquid systems"*, Chemical Engineering Science (2023)
- **Capability:** Instance segmentation — assigns individual pixel masks to each droplet
- **Advantage over YOLO:** Precise boundary → exact diameter and spreading diameter from contour

#### DropletMask

- **Paper:** *"DropletMask: Leveraging visual data for droplet impact analysis"*, Droplet, Wiley (2024)
- **Specifically designed** for droplet impact video analysis
- Uses Mask R-CNN for cross-frame tracking
- **This is the closest published work to exactly what you need**

#### Segment Anything Model (SAM, Meta AI, 2023)

- Zero-shot segmentation — no training needed on your specific data
- Provide a point prompt inside the droplet; SAM returns the full segmented mask
- Works well for the high-contrast droplet-on-background in your shadowgraphy images
- Available: `pip install segment-anything` + model weights from Meta

```python
# Example SAM usage for one frame
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(frame_rgb)
# Click on droplet centroid
masks, _, _ = predictor.predict(point_coords=np.array([[cx, cy]]),
                                 point_labels=np.array([1]))
droplet_mask = masks[0]  # pixel-level mask
```

---

### 6.3 Deep Learning Super-Resolution for Sub-pixel Accuracy

#### 4S-SROF (4-Segment Super-Resolution Optimized Fitting)

- **Paper:** *"Deep Learning to Analyze Sliding Drops"*, Langmuir, ACS (2023)
- **Method:** Super-resolution upscaling (ratio 3×, PSNR = 36.39 dB) before geometric fitting
- **Accuracy gain:** 21% improvement for contact angles <90°; 33% improvement >90°
- **Relevance:** Apply before measuring droplet diameter and spreading — improves sub-pixel accuracy, especially important when droplet is small relative to frame size

---

### 6.4 Physics-Informed Neural Networks (PINNs)

#### PINNs4Drops

- **Paper:** *"PINNs4Drops: Video-conditioned physics-informed neural networks for two-phase flow reconstruction"*, arXiv:2411.15949 (November 2024)
- **Method:** Combines Navier-Stokes equations with experimental video frames. The neural network is trained to satisfy both the governing physics and the observed droplet boundary from video
- **Output:** Full 3D velocity and pressure field reconstruction from 2D video
- **Relevance:** Advanced — useful for reconstructing the internal flow dynamics of your droplet during impact, not just surface observables. Most valuable if you want to publish mechanistic insights beyond just empirical metrics

---

### 6.5 Machine Learning Regression for Parameter Prediction

These methods take the dimensionless numbers ($We$, $Re$, $Oh$, contact angle) as inputs and predict output metrics directly — useful for building a predictive model from your dataset.

#### Gaussian Process Regression for Spreading & Rebound

- **Paper:** *"Investigation of Droplet Spreading and Rebound Dynamics on Superhydrophobic Surfaces Using Machine Learning"*, MDPI Micromachines, Vol. 10(6):357 (2025)
- **Dataset:** 1498 droplet impact experiments
- **Inputs:** $We$, $Re$, surface properties
- **Outputs:** Maximum spreading coefficient $\beta_{max}$, contact time, rebound efficiency
- **Finding:** Isotropic exponential Gaussian process regression **outperforms all published empirical correlations**
- **Relevance:** After you extract all parameters, train this type of model on your ~60–100 impact events to predict behavior for new sample types

#### Support Vector Regression + ANN Classification

- **Paper:** *"Predicting Impact Outcomes and Maximum Spreading of Drop Impact on Heated Nanostructures Using Machine Learning"*, Langmuir, ACS (2023)
- **Method:** SVR for spreading prediction; ANN classifier for outcome (bounce/stick/splash)
- **Accuracy:** Up to 98% for binary outcome classification
- **Relevance:** Classify each of your samples into bouncing vs. partial wetting vs. full wetting categories automatically

#### NARX-ANN for Temporal Spreading Dynamics

- **Paper:** *"Prediction of droplet spreading dynamics: NARX-ANN approach"*, Chemical Engineering Research and Design (2020)
- **Method:** Nonlinear Auto-Regressive eXogenous Artificial Neural Network — handles **irregular sampling intervals** from video
- **Inputs:** $We$, Ohnesorge number $Oh = \mu / \sqrt{\rho \sigma D_0}$, equilibrium contact angle
- **Output:** Full spreading diameter curve $D(t)$

---

### 6.6 Generative AI for Training Data Augmentation

If you annotate only a small number of frames, GAN-based augmentation can expand your training set.

#### DropletGAN

- **Paper:** *"Enhanced Droplet Analysis Using Generative Adversarial Networks"*, arXiv:2402.15909 (2024)
- **Method:** Progressive GAN trained on small high-speed camera dataset → generates realistic 1024×1024 droplet images (FID score = 11.29)
- **Result:** 16% increase in detection mAP when YOLO trained on synthetic + real vs. real only
- **Relevance:** If your annotated dataset is small (<500 frames), use GAN augmentation before training YOLO

#### BYG-drop (Blender + YOLO + CycleGAN)

- **Paper:** *"BYG-drop: enhanced droplet detection through machine learning and synthetic imaging"*, Frontiers in Chemical Engineering (2024)
- **Method:** Generate 3D synthetic droplet scenes in Blender → apply CycleGAN to match real image style → train YOLOv5
- **Performance:** 2.3 s/image processing, recall 2× that of competing tools

---

### 6.7 Recurrent and Transformer Models for Temporal Tracking

#### RNN for Sliding Drop Width Estimation

- **Paper:** *"Estimating sliding drop width via side-view features using recurrent neural networks"*, Scientific Reports (2024)
- **Method:** LSTM/GRU processes the sequence of frame features to predict droplet width over time
- **Relevance:** Can predict spreading diameter as a time series directly from video features — no explicit segmentation needed

#### CNN-Transformer Hybrid

- **Paper:** *"CNN-Transformer with Absolute Positional Encoding for Low-Dimensional Inputs: Applied to Drop Width Estimation"*, Springer (2023–2024)
- **Method:** CNN extracts per-frame features; Transformer captures temporal dependencies across frames
- **Advantage:** Better temporal coherence than frame-by-frame measurement — reduces noise in velocity and diameter time series

---

### 6.8 Summary Table — AI Methods

| Method | Best for | Training needed? | Accuracy |
|---|---|---|---|
| YOLOv8 + DeepSORT | Tracking, velocity | Yes (~200 annotated frames) | High |
| Mask R-CNN / DropletMask | Diameter, spreading | Yes | High |
| SAM (Segment Anything) | Diameter, spreading | **No** (zero-shot) | Medium–High |
| 4S-SROF super-resolution | Sub-pixel diameter accuracy | Pretrained available | +21–33% |
| Gaussian Process Regression | Predict $\beta_{max}$, $e$ | No (use your extracted data) | Outperforms empirical models |
| SVR + ANN Classifier | Outcome classification | No (use your data) | Up to 98% |
| NARX-ANN | Full spreading curve | No (use your data) | Good |
| PINNs4Drops | Flow field reconstruction | Yes (per-experiment) | Physically consistent |
| DropletGAN / BYG-drop | Data augmentation | Yes (small seed set) | +16% mAP |

---

## 7. Key Dimensionless Numbers

All parameters should be reported alongside these dimensionless groups to enable comparison with literature.

| Number | Formula | Physical meaning |
|---|---|---|
| Weber ($We$) | $\rho v^2 D_0 / \sigma$ | Inertia vs. surface tension |
| Reynolds ($Re$) | $\rho v D_0 / \mu$ | Inertia vs. viscosity |
| Ohnesorge ($Oh$) | $\mu / \sqrt{\rho \sigma D_0}$ | Viscosity vs. inertia+surface tension |
| Spreading factor | $\beta_{max} = D_{max}/D_0$ | Lateral deformation at impact |
| Coeff. of restitution | $e = v_{rebound}/v_{impact}$ | Energy retained after bounce |
| Contact time | $\tau^* = \tau \sqrt{\sigma / \rho R_0^3}$ | Dimensionless contact duration |

For pure water at room temperature: $\rho = 998\ \text{kg/m}^3$, $\sigma = 0.072\ \text{N/m}$, $\mu = 0.001\ \text{Pa·s}$  
For nanoparticle dispersions: $\sigma$ and $\mu$ will differ — measure or estimate from literature for each sample type.

**Expected ranges for superhydrophobic surfaces at 1.14 m/s, 2 mm droplet:**
- $We \approx 36$
- $Re \approx 2276$
- $Oh \approx 0.0027$
- $\beta_{max} \approx 2.0–3.5$ (varies strongly with sample)
- $e \approx 0.1–0.8$ (varies strongly with sample)

---

## 8. Recent Literature

### Directly Relevant Papers

| Paper | Year | Key contribution |
|---|---|---|
| Durve et al., *Physics of Fluids* 34, 082003 | 2022 | DropTrack: YOLOv5+DeepSORT for droplet tracking |
| Vaikuntanathan & Sivakumar, *Soft Matter* | 2016 | Surfactant concentration effects on droplet impact on SH surfaces |
| Pack et al., *Langmuir* | 2017 | Nanoparticle-laden droplets on superhydrophobic surfaces |
| Liu et al., *Nature Physics* | 2014 | Pancake bouncing on SH surfaces — contact time reduction |
| Antonini et al., *Langmuir* | 2013 | Contact time and rebound on SH surfaces |
| Pepper et al. | 2008 | Dynamic vs. equilibrium surface tension at impact timescales |
| Yarin, *Annual Review Fluid Mech* | 2006 | Comprehensive review of droplet impact on dry surfaces |

### AI/ML Papers for Droplet Analysis

| Paper | Year | Method | Key finding |
|---|---|---|---|
| *"Droplet Spreading/Rebound ML"*, MDPI Micromachines | 2025 | Gaussian Process Regression | Outperforms all published empirical correlations for $\beta_{max}$ |
| *"DropTrack"*, Physics of Fluids | 2022 | YOLOv5 + DeepSORT | Real + synthetic training gives best accuracy |
| *"PINNs4Drops"*, arXiv 2411.15949 | 2024 | Video-conditioned PINNs | Full 3D flow reconstruction from 2D video |
| *"Deep Learning for Sliding Drops"*, Langmuir | 2023 | 4S-SROF super-resolution | 21–33% accuracy improvement in contact angle |
| *"DropletMask"*, Droplet (Wiley) | 2024 | Mask R-CNN | Automated droplet impact video analysis |
| *"Predicting Impact on Heated Surfaces"*, Langmuir | 2023 | SVR + ANN | 98% accuracy for outcome classification |
| *"DropletGAN"*, arXiv 2402.15909 | 2024 | GAN augmentation | +16% mAP with synthetic data |
| *"BYG-drop"*, Frontiers Chem. Eng. | 2024 | Blender+YOLO+CycleGAN | 2× recall vs. competing tools |
| *"Spreading on supercooled substrates"*, Colloids Surf. A | 2023 | CNN with $We$, $Re$, $Ca$ inputs | Spreading dynamics prediction |
| *"NARX-ANN for spreading dynamics"*, Chem. Eng. R&D | 2020 | NARX-ANN | Handles irregular video sampling |
| *"ML contact angle measurement"*, Sci. Reports | 2023 | CNN goniometry | Stable under blur and illumination variation |
| *"Benchmarking YOLO for droplets"*, Eur. Phys. J. E | 2023 | YOLOv5 vs YOLOv7 | YOLOv7 10% faster; GPU needed for real-time |
| *"RNN for sliding drop width"*, Sci. Reports | 2024 | LSTM/GRU | Width prediction from side-view video sequence |
| *"CNN-Transformer for drop width"*, Springer | 2024 | CNN + Transformer | Better temporal coherence than per-frame methods |

### Open-Source Tools

| Tool | Method | Use case | Link |
|---|---|---|---|
| **Segment Anything (SAM)** | ViT-based | Zero-shot droplet segmentation | github.com/facebookresearch/segment-anything |
| **DropTrack** | YOLOv5+DeepSORT | Droplet tracking & trajectory | ar5iv.labs.arxiv.org/html/2205.02568 |
| **ADM Software** | OpenCV | Automated droplet measurement | a-d-m.weebly.com |
| **FluoroCellTrack** | Python/OpenCV | High-throughput droplet analysis | github.com/Manibarathi/FluoroCellTrack |
| **CellProfiler** | GUI pipeline | Batch droplet image analysis | github.com/taltechmicrofluidics/CP-for-droplet-analysis |
| **Bonsai (Lab on Chip)** | Visual programming | Real-time microfluidic analysis | RSC Lab on a Chip 2023 |

---

## 9. Recommended Workflow

### Phase 1: Calibration (do once)

```
1. Extract frame from scale.mp4 (horizontal ruler)
2. Measure pixel distance between 1 mm tick marks → px_per_mm
3. Extract frame from scale v.mp4 (vertical ruler) → px_per_mm_vertical
4. Confirm actual camera FPS from operator notes or camera software
```

### Phase 2: Classical CV extraction (for all videos)

```
For each .mp4 file:
  1. Find impact frame (max frame-diff in bottom region)
  2. Pre-impact: HoughCircles on frames [impact-12 : impact-2]
     → centroid positions → linear fit → v_initial
     → median radius → D0
  3. During contact: scan frames [impact : impact+25]
     → max horizontal width at surface row → D_max
  4. Post-bounce: HoughCircles on frames [liftoff : liftoff+8]
     → centroid positions → linear fit → v_rebound
  5. Compute We, Re, Oh, e, beta_max
  6. Save to CSV
```

### Phase 3: AI-assisted extraction (for precision & automation)

```
Option A (fastest, no training):
  - Use SAM (Segment Anything) for pixel-accurate segmentation
  - Point-prompt on droplet centroid in each frame
  - Extract contour → diameter, spreading width

Option B (most robust for batch processing):
  - Annotate 200-300 frames using CVAT or LabelImg
  - Fine-tune YOLOv8 on your dataset
  - Run DeepSORT tracking → full trajectories for all videos

Option C (highest scientific value):
  - After extracting all parameters with A or B
  - Train Gaussian Process Regression model
  - Input: We, Re, Oh, sample type
  - Output: predicted beta_max, e
  - Validate on held-out replicates
```

### Phase 4: Statistical analysis

```
For each sample group (minimum 3 replicates):
  - Mean ± standard deviation of: v_initial, v_rebound, D0, D_max, e, beta_max
  - ANOVA or t-test between sample groups
  - Correlation plots: e vs. We, beta_max vs. We
  - Group by: particle morphology (above/below CMC synthesis), surfactant type, surfactant present/absent
```

---

## 10. Next Steps

### Immediate (before any analysis)

- [ ] **Confirm exact camera FPS** — contact the operator or check Photron/Phantom software export log. Physics estimate: ~4000 fps. All velocities scale directly with this number.
- [x] **Calibrate scale** — completed. `px_per_mm = 65.625` (horizontal, measured from 17 ticks / 16 mm). Vertical pending physical verification of tick spacing (see Section 3.2).
- [x] **Identify surface y-row** — completed. Auto-detected per video using horizontal Sobel edge detection. Surface row varies between recordings (slide was repositioned between sessions). Use `detect_surface_row()` function per video — do NOT hardcode a single value.

### Short term

- [ ] Build automated impact frame finder and test on `water.mp4` through `water6.mp4` — these have known behavior (pure water, should bounce well) and serve as ground truth.
- [ ] Extract all 4 parameters for all water replicates; verify $v_{impact} \approx 1.14\ \text{m/s}$ and $D_0 \approx 2.0\ \text{mm}$.
- [ ] Run SAM on representative frames to compare with HoughCircles accuracy.

### Medium term

- [ ] Process all 02182026 and 03242026 videos → master CSV with all parameters per video.
- [ ] Train YOLOv8 on annotated subset for robust automated tracking.
- [ ] Train Gaussian Process or SVR regression on extracted dataset.

### Long term

- [ ] Compare sample groups: effect of surfactant type, CMC level, particle morphology (washed vs. unwashed).
- [ ] Compute Weber number for each sample — note that $\sigma$ changes with surfactant concentration; measure equilibrium surface tension for each formulation.
- [ ] Consider PINNs4Drops for one or two representative cases to get full flow field analysis for publication.

---

*Documentation generated: 2026-04-12*  
*Working directory: `c:/Users/User/Desktop/Materials/`*
