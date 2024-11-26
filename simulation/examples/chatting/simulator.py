from datetime import timedelta
import math
import os
import random
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import dill
import time
from concurrent import futures
from loguru import logger

import agentscope
from agentscope.agents.agent import DistConf

from simulation.helpers.constants import *
from agentscope.constants import _DEFAULT_SAVE_DIR
from simulation.examples.chatting.agent import *
from simulation.examples.chatting.environment.env import ChatRoom
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *
from simulation.helpers.base_simulator import BaseSimulator

CUR_ROUND = 1

scene_path = os.path.dirname(os.path.abspath(__file__))


class Simulator(BaseSimulator):
    def __init__(self):
        super().__init__(scene_path=scene_path)

    def _init_agentscope(self):
        agentscope.init(
            project=self.config["project_name"],
            save_code=False,
            save_api_invoke=False,
            model_configs=os.path.join(
                scene_path, CONFIG_DIR, self.config["model_configs_path"]
            ),
            use_monitor=False,
            save_dir=os.path.join(_DEFAULT_SAVE_DIR, self.config["project_name"]),
            runtime_id=self.config["runtime_id"],
        )

    def _create_envs(self, agent_num):
        logger.info("Init environment")
        embedding_api_num = len(self.config["embedding_api"])
        env_num = math.ceil(agent_num / AGENT_PER_ENV)
        env_names = [f"environment-{str(i)}" for i in range(env_num)]
        env_ports = [
            i % self.config["server_num_per_host"] + self.config["base_port"]
            for i in range(env_num)
        ]
        ann = Msg(
            name="Boss",
            content=(
                "This is a game development work group, "
                "please discuss how to develop an open world game."
            ),
            role="system",
        )
        envs = []
        with futures.ThreadPoolExecutor() as executor:
            args = [
                {
                    "name": name,
                    "embedding_api": self.config["embedding_api"][
                        i % embedding_api_num
                    ],
                    "announcement": ann,
                    "to_dist": DistConf(host=self.config["host"], port=port),
                }
                for i, name, port in zip(range(len(env_names)), env_names, env_ports)
            ]
            for env in tqdm(
                executor.map(lambda arg: ChatRoom(**arg), args),
                total=len(env_names),
                desc="Init environments",
            ):
                envs.append(env)

        self.envs = envs
        self.env = envs[0]

    def _create_agents(self, agent_configs):
        logger.info(f"Init {len(agent_configs)} chatting agents")
        agents = []
        with futures.ThreadPoolExecutor() as executor:
            args = [
                {
                    "env": self.env,
                    **config["args"],
                    "to_dist": DistConf(
                        host=config["args"]["host"], port=config["args"]["port"]
                    ),
                }
                for config in agent_configs
            ]
            for agent in tqdm(
                executor.map(lambda arg: ChatRoomAgent(**arg), args),
                total=len(agent_configs),
                desc="Init agents",
            ):
                agents.append(agent)

        self.agents = agents
        return agents

    def _init_agents_envs(self):
        # Prepare agents args
        agent_configs = self._prepare_agents_args()
        # Init envs
        self._create_envs(len(agent_configs))
        # Resume envs
        if self.resume:
            logger.info("Resume envs...")
            results = []
            for env, state in zip(self.envs, self.env_save_state):
                results.append(env.load(state))
                env.agent_id = dill.loads(state)["_oid"]
            for res in tqdm(results, total=len(results), desc="Resume envs"):
                res.result()

        # Init agents
        agents = self._create_agents(agent_configs)
        # Resume agents
        if self.resume:
            logger.info("Resume agents...")
            results = []
            for agent, state in zip(agents, self.agent_save_state):
                results.append(agent.load(state))
                agent.agent_id = dill.loads(state)["_oid"]
            for res in tqdm(results, total=len(results), desc="Resume agents"):
                res.result()

        # Set all_agents for envs
        self._set_env4agents()

    def run(self):
        for r in range(self.cur_round, self.config["round_n"] + 1):
            logger.info(f"Round {r} started")
            self._one_round()
            self.env.chatting()
            self.save()

        self.cur_round = -1
        logger.info("Simulation finished")

    def get_save_state(self):
        results = []
        for agent in self.agents:
            results.append(agent.save())
        agent_save_state = []
        for res in tqdm(results, total=len(results), desc="Get agent save state"):
            agent_save_state.append(res.result())

        results = []
        for env in self.envs:
            results.append(env.save())
        env_save_state = []
        for res in tqdm(results, total=len(results), desc="Get env save state"):
            env_save_state.append(res.result())

        return agent_save_state, env_save_state


if __name__ == "__main__":
    start_time = time.time()
    simulator = Simulator()
    end_time = time.time()
    formatted_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f"Init Agent Total time: {formatted_time}")

    start_time = time.time()
    simulator.run()
    end_time = time.time()
    formatted_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f"Simulation Total time: {formatted_time}")
