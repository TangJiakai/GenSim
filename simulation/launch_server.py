import importlib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import argparse

import agentscope
from agentscope.server import RpcAgentServerLauncher

from simulation.helpers.constants import *
from simulation.helpers.utils import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base_port", type=int, default=13000)
    parser.add_argument("--scenario", type=str, required=True)
    return parser.parse_args()


def setup_participant_agent_server(host: str, port: int, scenario: str) -> None:
    scene_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "examples", scenario
    )

    config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))

    try:
        module = importlib.import_module(
            f"simulation.examples.{config['project_name']}.agent"
        )
        globals().update(vars(module))
    except Exception as e:
        raise e

    """Set up agent server"""
    agentscope.init(
        project=config["project_name"],
        name="server",
        runtime_id=str(port),
        save_code=False,
        save_api_invoke=False,
        model_configs=os.path.join(scene_path, CONFIG_DIR, MODEL_CONFIG),
        use_monitor=False,
    )
    assistant_server_launcher = RpcAgentServerLauncher(
        host=host,
        port=port,
        # pool_type="redis",
        max_pool_size=8192000000,
        max_expire_time=7200000,
    )
    assistant_server_launcher.launch(in_subprocess=False)
    assistant_server_launcher.wait_until_terminate()


if __name__ == "__main__":
    args = parse_args()
    setup_participant_agent_server(args.host, args.base_port, args.scenario)
