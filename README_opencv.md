# Droplet Segmentation — OpenCV Script

Detects and tracks droplets frame-by-frame using background subtraction and contour analysis.
No GPU required. Runs on CPU in seconds to minutes depending on video length.

---

## How it works

1. **Background model** — averages the first N frames (before the droplet enters) to build a static background image.
2. **Foreground detection** — subtracts background from each frame and thresholds the difference image.
3. **Morphological cleanup** — close small holes, open stray noise.
4. **Contour filtering** — keeps only contours that pass:
   - minimum area (noise filter)
   - circularity (rejects non-droplet shapes)
   - fully inside the frame (rejects partially-cut droplets at the edges)
5. **Reference frame** — the first frame where at least one droplet passes all filters becomes the reference. The total contour area in that frame = 100%.
6. **Per-frame output** — for every subsequent frame, each detected droplet gets a row: centroid, area, and percentage of the reference area.

---

## Installation

```bash
pip install opencv-python numpy
```

---

## Usage

```bash
python analyze_droplet_opencv.py --video cainhsds1.mp4
```

Full options:

```bash
python analyze_droplet_opencv.py \
  --video       cainhsds1.mp4 \
  --output      droplet_results_opencv.csv \
  --bg_frames   30 \
  --diff_thresh 25 \
  --min_area    150 \
  --circ_thresh 0.3 \
  --margin      3
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--video` | `cainhsds1.mp4` | Input video file |
| `--output` | `droplet_results_opencv.csv` | Output CSV file |
| `--bg_frames` | `30` | How many leading frames to use for the background model. Must be frames **before** the droplet enters. Increase if the droplet enters very late; decrease if the video has almost no empty frames at the start. |
| `--diff_thresh` | `25` | Pixel-level brightness difference needed to classify a pixel as foreground. Lower = more sensitive (picks up faint droplets, but more noise). Higher = less sensitive (cleaner, but may miss dim droplets). |
| `--min_area` | `150` | Minimum contour area in pixels. Rejects small noise blobs. Set this to roughly half the expected droplet area in pixels. |
| `--circ_thresh` | `0.3` | Circularity score (0–1). A perfect circle = 1.0. After impact the droplet deforms, so keep this low (0.2–0.4). Set to 0 to disable. |
| `--margin` | `3` | Pixels from the frame edge. A contour touching within this margin is considered partially outside and ignored (used to find the first **fully** visible frame). |

---

## Output CSV format

```
frame, drop_id, cx, cy, area_px, percentage
0042,  1,       318, 144, 1820,  100.0
0091,  1,       317, 290, 2105,  100.0
0105,  1,       316, 318, 1530,  84.1
0105,  2,       341, 302,  290,  15.9
```

| Column | Description |
|---|---|
| `frame` | Frame index (0-based) |
| `drop_id` | 1 = largest fragment, 2 = second largest, etc. |
| `cx`, `cy` | Centroid x and y in pixels |
| `area_px` | Contour area in pixels |
| `percentage` | Area as % of the reference droplet area |

---

## Tuning tips

- **Droplet not detected at all** → lower `--diff_thresh` and `--min_area`.
- **Too many false detections** → raise `--diff_thresh`, raise `--min_area`, raise `--circ_thresh`.
- **Reference frame set too late** (droplet partially in frame) → lower `--margin` to 1.
- **Reference frame set too early** (droplet partially outside at top) → raise `--margin` to 5–10.
- **Background not static in your video** → increase `--bg_frames` to capture more of the empty scene, or manually inspect which frames are droplet-free and set `--bg_frames` accordingly.
- **Post-impact the droplet spreads flat and loses circularity** → set `--circ_thresh 0.1` or `0.0` to keep deformed shapes.

---

## Limitations

- Background subtraction assumes the first `--bg_frames` frames contain no droplet. If the droplet enters immediately in frame 0, this will fail (use SAM2 script instead).
- Does not track droplet identity across frames (no ID linking between frame N and frame N+1). Each frame is processed independently.
- If two droplets overlap their contours may merge into one. This is physically unlikely in this experiment but worth checking around the impact moment.
