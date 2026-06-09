#!/bin/bash
# SAM2 batch for new_experiments — px/mm: 05112026=66.5, 05122026=56.0
# Long videos (>5000 frames) use frame_step=4

CKPT="/data/checkpoints/sam2.1_hiera_large.pt"
CFG="configs/sam2.1/sam2.1_hiera_l.yaml"

VDIR1="/home/ubuntu/materials/new_experiments/05112026"
OUT1="/home/ubuntu/materials/new_experiments/05112026_sam2_results"

VDIR2="/home/ubuntu/materials/new_experiments/05122026"
OUT2="/home/ubuntu/materials/new_experiments/05122026_sam2_results"

mkdir -p "$OUT1" "$OUT2"

cd /home/ubuntu/materials

run() {
    local video="$1"
    local out_csv="$2"
    local step="${3:-1}"
    local stem=$(basename "$video" .mp4)
    if [ -f "$out_csv" ]; then
        echo "[skip] $stem (already done)"
        return
    fi
    echo ">>> $stem  (frame_step=$step)"
    /data/venv/bin/python analyze_droplet_sam2.py \
        --video "$video" \
        --output "$out_csv" \
        --checkpoint "$CKPT" \
        --model_cfg "$CFG" \
        --frame_step "$step"
}

echo "=== 05112026 ==="
run "$VDIR1/nr50water4.mp4"            "$OUT1/nr50water4_sam2.csv"     1
run "$VDIR1/water 3.mp4"               "$OUT1/water 3_sam2.csv"        1
run "$VDIR1/ca only 2.mp4"             "$OUT1/ca only 2_sam2.csv"      4
run "$VDIR1/ca only 3.mp4"             "$OUT1/ca only 3_sam2.csv"      4
run "$VDIR1/nr50water.mp4"             "$OUT1/nr50water_sam2.csv"      4
run "$VDIR1/nr50water2.mp4"            "$OUT1/nr50water2_sam2.csv"     4
run "$VDIR1/nr50water3.mp4"            "$OUT1/nr50water3_sam2.csv"     4
run "$VDIR1/water 2.mp4"               "$OUT1/water 2_sam2.csv"        4

echo ""
echo "=== 05122026 ==="
run "$VDIR2/0.028tx1.mp4"              "$OUT2/0.028tx1_sam2.csv"       1
run "$VDIR2/0.45sds1.mp4"              "$OUT2/0.45sds1_sam2.csv"       1
run "$VDIR2/cain0.45sds.mp4"           "$OUT2/cain0.45sds_sam2.csv"    1
run "$VDIR2/0.028tx2.mp4"              "$OUT2/0.028tx2_sam2.csv"       4
run "$VDIR2/0.028tx3.mp4"              "$OUT2/0.028tx3_sam2.csv"       4
run "$VDIR2/0.45sds2.mp4"              "$OUT2/0.45sds2_sam2.csv"       4
run "$VDIR2/0.45sds3.mp4"              "$OUT2/0.45sds3_sam2.csv"       4
run "$VDIR2/cain0.028tx1.mp4"          "$OUT2/cain0.028tx1_sam2.csv"   4
run "$VDIR2/cain0.028tx2.mp4"          "$OUT2/cain0.028tx2_sam2.csv"   4
run "$VDIR2/cain0.08cg1.mp4"           "$OUT2/cain0.08cg1_sam2.csv"    4
run "$VDIR2/cain0.08cg2.mp4"           "$OUT2/cain0.08cg2_sam2.csv"    4
run "$VDIR2/cain0.08cg3.mp4"           "$OUT2/cain0.08cg3_sam2.csv"    4
run "$VDIR2/cain0.45sds2.mp4"          "$OUT2/cain0.45sds2_sam2.csv"   4
run "$VDIR2/cain0.45sds3.mp4"          "$OUT2/cain0.45sds3_sam2.csv"   4
# cain0.028tx3.mp4 is corrupted (0 frames) — skipped

echo ""
echo "=== new_experiments SAM2 batch complete ==="
echo "05112026:"; ls "$OUT1"/*.csv 2>/dev/null | wc -l
echo "05122026:"; ls "$OUT2"/*.csv 2>/dev/null | wc -l
