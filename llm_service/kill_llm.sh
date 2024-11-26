#!/bin/bash

current_dir="$(cd "$(dirname "$0")"; pwd)"
PID_FILE="$current_dir/.pid"
LOG_FILE="$current_dir/.log"

if [ -f "$PID_FILE" ]; then
    while IFS= read -r PID; do
        if kill -0 "$PID" 2>/dev/null; then 
            kill -9 "$PID"
            echo "Process $PID has been terminated."
        else
            echo "Process $PID does not exist."
        fi
    done < "$PID_FILE"
    rm "$PID_FILE"
else
    echo "No PID file found."
fi

rm "$LOG_FILE"
