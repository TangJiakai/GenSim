#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

current_dir=$(cd `dirname $0`; pwd)
parent_dir=$(cd `dirname $0`/..; pwd)

LLM_FILE="your_llm_path"

LOG_FILE="$current_dir/.log"
PID_FILE="$current_dir/.pid"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tuning-mode>"
    exit 1
fi

python "$parent_dir/code/tune_llm.py" --tuning_mode $1 --llm_path $LLM_FILE 2>> $LOG_FILE & 

echo $! >> $PID_FILE

echo "LLM tuning is done."