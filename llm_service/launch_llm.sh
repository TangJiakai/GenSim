#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: $0 <port> <gpu_id>"
    exit 1
fi

port=$1
gpuid=$2
export CUDA_VISIBLE_DEVICES="$gpuid"
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "Port: $port"
echo "GPU ID: $CUDA_VISIBLE_DEVICES"

current_dir=$(cd `dirname $0`; pwd)
llm_tuning_dir="$(dirname "$current_dir")/llm_tuning/saves"

LLM_FILE="your_llm_path"

LOG_FILE="${current_dir}/.log"
PID_FILE="${current_dir}/.pid"

if [ -f "${llm_tuning_dir}/adapter_config.json" ]; then
    echo "Using adapter-based LORA." >> $LOG_FILE
    python -m vllm.entrypoints.openai.api_server \
        --model $LLM_FILE \
        --trust-remote-code \
        --port $port \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --lora-modules lora="${llm_tuning_dir}" \
        --disable-frontend-multiprocessing \
        --guided-decoding-backend=lm-format-enforcer \
        --gpu-memory-utilization 0.8 \
        2>> $LOG_FILE &
else
    python -m vllm.entrypoints.openai.api_server \
        --model $LLM_FILE \
        --trust-remote-code \
        --port $port \
        --dtype auto \
        --pipeline-parallel-size 1 \
        --enforce-eager \
        --enable-prefix-caching \
        --enable-lora \
        --disable-frontend-multiprocessing \
        --guided-decoding-backend=lm-format-enforcer \
        --gpu-memory-utilization 0.9 \
        2>> $LOG_FILE &
fi

echo $! >> $PID_FILE

sleep 10

echo "LLM API is running."