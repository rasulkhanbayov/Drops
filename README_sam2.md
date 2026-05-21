# Droplet Segmentation — SAM2 Script

Detects and tracks droplets using Meta's Segment Anything Model 2 (SAM2) video predictor.
**Requires a CUDA GPU.** Produces more accurate masks than the OpenCV script, especially around impact and fragmentation.

---

## How it works

1. **Auto-locate reference frame** — uses a fast OpenCV background subtraction pass to find the first frame where the droplet is fully visible, and records the droplet's centroid.
2. **Extract frames** — saves every video frame as a JPEG to a temporary directory (SAM2 video predictor requires image files).
3. **SAM2 point prompt** — feeds the detected centroid as a foreground point into SAM2's video predictor at the reference frame.
4. **Mask propagation** — SAM2 propagates the segmentation mask forward through all remaining frames, maintaining coherent tracking even through deformation and partial occlusion.
5. **Split detection** — after impact the propagated mask may contain multiple disconnected regions (fragments). Connected-components analysis splits the mask into individual drops automatically.
6. **Reference area** — the pixel area of the SAM2 mask on the reference frame = 100%. All subsequent areas are expressed relative to this.
7. **CSV output** — centroid, area, and percentage written for every drop in every frame.

---

## Requirements

### Hardware
- NVIDIA GPU with at least **8 GB VRAM** (16 GB recommended for large model)
- CUDA 11.8 or higher

### Python environment

```bash
# Create a fresh conda or venv environment (Python 3.10+)
conda create -n sam2_droplet python=3.10
conda activate sam2_droplet
```

### Install PyTorch (match your CUDA version)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Install SAM2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

Or clone and install locally (recommended for access to config files):

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### Install other dependencies

```bash
pip install opencv-python numpy
```

---

## Download SAM2 checkpoint

Four model sizes are available. Larger = more accurate but slower.

```bash
mkdir -p checkpoints
cd checkpoints

# Large (recommended — best accuracy)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Base+ (good balance)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Tiny (fastest, least accurate)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

On Windows, use `curl` instead of `wget`:

```powershell
curl -o checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

---

## Config file paths

The `--model_cfg` argument expects the path **relative to the sam2 package root** (i.e. where `sam2/` directory lives). If you cloned the repo locally, run the script from inside that directory, or pass the absolute path.

| Model | Config path |
|---|---|
| Large | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| Base+ | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| Small | `configs/sam2.1/sam2.1_hiera_s.yaml` |
| Tiny | `configs/sam2.1/sam2.1_hiera_t.yaml` |

---

## Usage

Basic (place the script and video in the same folder as the `checkpoints/` dir):

```bash
python analyze_droplet_sam2.py --video cainhsds1.mp4
```

Full options:

```bash
python analyze_droplet_sam2.py \
  --video       cainhsds1.mp4 \
  --output      droplet_results_sam2.csv \
  --checkpoint  checkpoints/sam2.1_hiera_large.pt \
  --model_cfg   configs/sam2.1/sam2.1_hiera_l.yaml \
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
| `--output` | `droplet_results_sam2.csv` | Output CSV file |
| `--checkpoint` | `checkpoints/sam2.1_hiera_large.pt` | SAM2 checkpoint `.pt` file |
| `--model_cfg` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM2 YAML config |
| `--bg_frames` | `30` | Frames for background model (OpenCV pre-pass to find reference frame) |
| `--diff_thresh` | `25` | Foreground pixel threshold for reference frame detection |
| `--min_area` | `150` | Minimum droplet area in pixels for initial detection |
| `--circ_thresh` | `0.3` | Minimum circularity for initial detection (0 = disabled) |
| `--margin` | `3` | Edge margin for fully-in-frame check |

---

## Output CSV format

```
frame, drop_id, cx, cy, area_px, percentage
0042,  1,       318, 144, 2104,  100.0
0091,  1,       317, 290, 2280,  100.0
0105,  1,       315, 319, 1762,  83.7
0105,  2,       342, 303,  342,  16.3
0106,  1,       314, 322, 1690,  80.3
0106,  2,       345, 299,  415,  19.7
```

| Column | Description |
|---|---|
| `frame` | Frame index (0-based) |
| `drop_id` | 1 = largest fragment, 2 = second largest, etc. |
| `cx`, `cy` | Centroid x and y in pixels |
| `area_px` | Mask area in pixels |
| `percentage` | Area as % of the SAM2 reference mask area |

---

## Tuning tips

- **Reference frame not found** → lower `--diff_thresh` and `--min_area`. The OpenCV pre-pass only needs to roughly locate the droplet; SAM2 will refine the mask.
- **SAM2 loses the droplet mid-video** → the model may need a negative prompt (background point). Consider adding a second point prompt to the background in the reference frame (requires modifying the script's `add_new_points_or_box` call with additional points and label `0`).
- **Post-impact fragments not separated** → lower the `split_min_area` variable in the script (line near `split_min_area = max(50, min_area // 3)`).
- **Out of VRAM** → switch to `sam2.1_hiera_small` or `sam2.1_hiera_tiny` checkpoints.
- **Very long video (thousands of frames)** → frame extraction will take significant disk space (~1–3 MB per frame). Ensure sufficient free disk space before running.

---

## Differences vs OpenCV script

| Feature | OpenCV | SAM2 |
|---|---|---|
| GPU required | No | Yes |
| Speed | Fast (seconds) | Slow (minutes–hours) |
| Mask quality | Contour-based, approximate | Pixel-accurate segmentation |
| Handles deformed shapes | Limited by circularity filter | Yes, fully |
| Tracks identity across frames | No (independent per frame) | Yes (coherent tracking) |
| Handles occlusion/blur | No | Partially |
| Split detection | Separate contours per frame | Connected components on SAM2 mask |
