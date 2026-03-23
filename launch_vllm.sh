#!/bin/bash
# Launch vLLM server for Qwen2.5-7B-Instruct
#
# Usage:
#   bash launch_vllm.sh                    # 默认单卡
#   CUDA_VISIBLE_DEVICES=0,1 bash launch_vllm.sh  # 指定GPU
#
# 启动后，在test_pipeline.py中设置:
#   BASE_URL = "http://localhost:8000/v1"
#   MODEL = "Qwen2.5-7B-Instruct"

MODEL_PATH="/data/hwt/hf_ckpt/Qwen2.5-7B-Instruct"
PORT=8000
HOST="0.0.0.0"
GPU=6 # 第一个参数指定GPU，默认GPU 0

export CUDA_VISIBLE_DEVICES=$GPU

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    echo "Please set MODEL_PATH to your local Qwen2.5-7B-Instruct path"
    exit 1
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  GPU:   $GPU"
echo "  Port:  $PORT"
echo "  API:   http://localhost:${PORT}/v1"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6
