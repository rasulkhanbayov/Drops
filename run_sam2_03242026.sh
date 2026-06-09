#!/bin/bash
# SAM2 batch for 03242026 — px/mm=65.625, same setup as 02182026
# Two videos need frame_step=4: ONLY CA cg ABOVE CMC3 (5520f) and ca+TR (16377f)

VDIR="/home/ubuntu/materials/03242026_particlesonlypreparedinsurfactant"
OUT="/home/ubuntu/materials/03242026_sam2_results"
CKPT="/data/checkpoints/sam2.1_hiera_large.pt"
CFG="configs/sam2.1/sam2.1_hiera_l.yaml"

cd /home/ubuntu/materials

run() {
    local video="$1"
    local step="${2:-1}"
    local stem=$(basename "$video" .mp4)
    local out_csv="$OUT/${stem}_sam2.csv"
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

# Standard videos (frame_step=1)
run "$VDIR/0.001percent cg.mp4"
run "$VDIR/0.028p.mp4"
run "$VDIR/0.028percrnt tx.mp4"
run "$VDIR/0.45percrnt sds.mp4"
run "$VDIR/ONLY CA SDS ABOVE CMC.mp4"
run "$VDIR/ONLY CA SDS ABOVE CMC1.mp4"
run "$VDIR/ONLY CA SDS ABOVE CMC2.mp4"
run "$VDIR/ONLY CA cg ABOVE CMC1.mp4"
run "$VDIR/ONLY CA cg ABOVE CMC2.mp4"
run "$VDIR/ONLY CA cg less CMC1.mp4"
run "$VDIR/ONLY CA cg less CMC2.mp4"
run "$VDIR/ONLY CA cg less CMC3.mp4"
run "$VDIR/ONLY CA sds less CMC1.mp4"
run "$VDIR/ONLY CA sds less CMC2.mp4"
run "$VDIR/ONLY CA tx ABOVE CMC1.mp4"
run "$VDIR/ONLY CA tx ABOVE CMC2.mp4"
run "$VDIR/ONLY CA tx ABOVE CMC3.mp4"
run "$VDIR/ONLY CA tx ABOVE CMC4.mp4"
run "$VDIR/ONLY CA tx less CMC1.mp4"
run "$VDIR/ONLY CA tx less CMC2.mp4"
run "$VDIR/ONLY CA tx less CMC3.mp4"

# Long videos (frame_step=4)
run "$VDIR/ONLY CA cg ABOVE CMC3.mp4" 4
run "$VDIR/ca+TR.mp4" 4

echo ""
echo "=== 03242026 SAM2 batch complete ==="
ls "$OUT"/*.csv | wc -l
