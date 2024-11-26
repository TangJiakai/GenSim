import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import argparse
import math
import json
from ruamel.yaml import YAML

from simulation.helpers.utils import load_json, load_yaml
from simulation.helpers.constants import *


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base_port", type=int, default=13000)
    parser.add_argument("--server_num_per_host", type=int, default=1)
    parser.add_argument("--scenario", type=str, default="job_seeking")
    return parser.parse_args()


def save_agent_configs(agent_configs, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(agent_configs, file, ensure_ascii=False, indent=4)


def update_simulation_config(scene_path, args):
    with open(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG), "r") as file:
        yaml = YAML()
        simulation_config = yaml.load(file)

    simulation_config["base_port"] = args.base_port
    simulation_config["server_num_per_host"] = args.server_num_per_host

    with open(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG), "w") as file:
        yaml.dump(simulation_config, file)


def main(args):
    host = args.host
    base_port = args.base_port
    server_num_per_host = args.server_num_per_host
    available_port_num = server_num_per_host

    scene_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "examples", args.scenario
    )
    simulation_config_path = os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG)

    update_simulation_config(scene_path, args)

    simulation_config = load_yaml(simulation_config_path)
    agent_configs_paths = [
        os.path.join(scene_path, CONFIG_DIR, agent_config_path)
        for agent_config_path in simulation_config["agent_configs_paths"].values()
    ]
    agent_configs = []
    for agent_configs_path in agent_configs_paths:
        agent_configs.append(load_json(agent_configs_path))
    agent_num = sum([len(agent_config) for agent_config in agent_configs])
    print("number of agent:", agent_num)

    agent_num_per_server = math.ceil(agent_num / available_port_num)
    print("agent_num_per_server:", agent_num_per_server)

    cnt = 0
    for agent_config in agent_configs:
        for agent in agent_config:
            agent["args"]["host"] = host
            agent["args"]["port"] = base_port + cnt % available_port_num
            cnt += 1

    for agent_configs_path, agent_config in zip(agent_configs_paths, agent_configs):
        save_agent_configs(agent_config, agent_configs_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
