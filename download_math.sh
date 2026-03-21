#!/bin/bash
# Download MATH dataset from HuggingFace mirror using huggingface-cli
#
# Usage:
#   bash download_math.sh
#
# Prerequisites:
#   pip install huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

DATASET="EleutherAI/hendrycks_math"
OUTPUT_DIR="/data/hwt/hf_data/math"

mkdir -p "$OUTPUT_DIR"

echo "Downloading MATH dataset from HF mirror..."
echo "Dataset: $DATASET"
echo "Mirror:  $HF_ENDPOINT"
echo "Output:  $OUTPUT_DIR"
echo ""

huggingface-cli download \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    "$DATASET"

echo ""
echo "Done. Files saved to $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
