#!/bin/bash

script_dir=$(cd "$(dirname "$0")"; pwd)

if [ ! -f "${script_dir}/.pid" ]; then
    echo "PID file not found. Are the servers running?"
    exit 1
fi

while read pid; do
    kill -9 $pid
    if [ $? -eq 0 ]; then
        echo "Killed server with PID $pid"
    else
        echo "Failed to kill server with PID $pid"
    fi
done < "${script_dir}/.pid"

rm -rf "${script_dir}/log"
rm "${script_dir}/.pid"

echo "All servers stopped."