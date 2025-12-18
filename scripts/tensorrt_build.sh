#!/bin/bash
# Build TensorRT-LLM engine pour FunctionGemma
# Usage: ./scripts/tensorrt_build.sh

set -e

MODEL_DIR="functiongemma-ha"
OUTPUT_DIR="trt-engine"
CHECKPOINT_DIR="trt-checkpoint"

echo "=== TensorRT-LLM Build pour FunctionGemma ==="
echo ""

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "Docker non trouvé. Installez Docker d'abord."
    exit 1
fi

# Vérifier nvidia-docker
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "nvidia-container-toolkit non trouvé."
    echo "Installez avec: sudo apt install nvidia-container-toolkit"
    exit 1
fi

echo "[1/4] Pull du container TensorRT-LLM..."
docker pull nvcr.io/nvidia/tensorrt-llm/release:0.14.0 || {
    echo "Essai avec latest..."
    docker pull nvcr.io/nvidia/tensorrt-llm:latest
}

echo ""
echo "[2/4] Conversion HuggingFace -> TensorRT-LLM checkpoint..."
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt-llm/release:0.14.0 \
    python3 -c "
from tensorrt_llm.models import GemmaForCausalLM
import torch

print('Chargement du modèle...')
model = GemmaForCausalLM.from_hugging_face(
    '${MODEL_DIR}',
    dtype='bfloat16',
)
print('Sauvegarde checkpoint...')
model.save_checkpoint('${CHECKPOINT_DIR}')
print('Checkpoint sauvegardé!')
"

echo ""
echo "[3/4] Build TensorRT engine..."
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt-llm/release:0.14.0 \
    trtllm-build \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --gemma_model_type gemma2 \
    --max_batch_size 1 \
    --max_input_len 512 \
    --max_seq_len 640 \
    --use_fused_mlp enable \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16

echo ""
echo "[4/4] Vérification..."
if [ -f "${OUTPUT_DIR}/rank0.engine" ]; then
    echo "=== Build réussi! ==="
    echo "Engine: ${OUTPUT_DIR}/rank0.engine"
    ls -lh ${OUTPUT_DIR}/
else
    echo "Erreur: Engine non trouvé"
    exit 1
fi
