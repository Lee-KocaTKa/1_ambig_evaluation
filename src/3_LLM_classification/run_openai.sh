#!/bin/bash
#SBATCH --job-name=evaclip_testing
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -e

# Models you want to run
MODELS=("gpt-4o")

# Default paths (edit if needed)
CAPTION_ROOT="../../../data/ViLStrUB/jsons_UNIT2/"
IMAGE_ROOT="../../../data/ViLStrUB/images/"

# Pass-through additional args (like --image-style)
EXTRA_ARGS="$@"

echo "▶ Running models: ${MODELS[@]}"

for MODEL in "${MODELS[@]}"; do
    echo
    echo "=============================="
    echo "🔵 Running model: $MODEL"
    echo "=============================="

    python openaimodels.py \
        --model-choice "$MODEL" \
        --caption-root "$CAPTION_ROOT" \
        --image-root "$IMAGE_ROOT" \
        $EXTRA_ARGS

    echo "✅ Finished $MODEL"
done

echo
echo "🎉 All models finished!"
