#!/bin/bash

script_path=$(cd `dirname $0`; pwd)
pid_file="$script_path/.pid"

if [[ -f $pid_file ]]; then
    while IFS= read -r pid; do
        if kill -0 $pid 2>/dev/null; then  
            kill $pid  
            echo "Killed process $pid"
        else
            echo "Process $pid does not exist"
        fi
    done < $pid_file
    rm $pid_file
else
    echo ".pid does not exist."
fi
