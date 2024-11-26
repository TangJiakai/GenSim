#!/bin/bash

script_dir=$(cd "$(dirname "$0")"; pwd)

# get number of server
if ! [[ "$1" =~ ^[0-9]+$ ]] || ! [[ "$2" =~ ^[0-9]+$ ]] || [[ -z "\$3" ]]; then
    echo "Usage: $0 <server_num_per_host> <base_port> <scenario>"
    exit 1
fi

server_num_per_host=$1
base_port=$2
scenario=$3

echo "server_num_per_host: $server_num_per_host"
echo "base_port: $base_port"
echo "scenario: $scenario"

mkdir -p log

> "${script_dir}/.pid"

for ((i=0; i<server_num_per_host; i++)); do
    port=$((base_port + i))
    python "${script_dir}/launch_server.py" --base_port ${port} --scenario "${scenario}" > log/${port}.log 2>&1 &
    echo $! >> "${script_dir}/.pid"
    echo "Started agent server on localhost:${port} with PID $!"
done

echo "All servers started."

python "${script_dir}/assign_host_port.py" --base_port ${base_port} --server_num_per_host ${server_num_per_host} --scenario "${scenario}"
echo "Assigned base ports to agent configs."
