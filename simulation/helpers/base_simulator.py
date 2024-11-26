import itertools
import math
import os
import dill
from tqdm import tqdm
from loguru import logger

import agentscope
from agentscope.manager import FileManager

from simulation.helpers.constants import *
from agentscope.constants import _DEFAULT_SAVE_DIR
from simulation.examples.recommendation.agent import *
from simulation.helpers.emb_service import *
from simulation.helpers.utils import *

CUR_ROUND = 1


class BaseSimulator:
    def __init__(self, scene_path):
        super().__init__()
        self.scene_path = scene_path
        self.config = load_yaml(os.path.join(scene_path, CONFIG_DIR, SIMULATION_CONFIG))
        self.cur_round = 1
        self.resume = False
        self.agent_save_state = None
        self._from_scratch()

    def _from_scratch(self):
        self._init_agentscope()

        if self.config["load_simulator_path"] is not None:
            logger.info(f"Load simulator from {self.config['load_simulator_path']}")
            config = self.config
            loaded_simulator = self.load(self.config["load_simulator_path"])
            self.__dict__.update(loaded_simulator.__dict__)
            self.config = config
            self.resume = True
            self._init_agents_envs()
            logger.info("Load simulator successfully")
        else:
            self._init_agents_envs()

        save_configs(self.config)

    def _init_agentscope(self):
        agentscope.init(
            project=self.config["project_name"],
            save_code=False,
            save_api_invoke=False,
            model_configs=os.path.join(self.scene_path, CONFIG_DIR, MODEL_CONFIG),
            use_monitor=False,
            save_dir=os.path.join(_DEFAULT_SAVE_DIR, self.config["project_name"]),
            runtime_id=self.config["runtime_id"],
        )

    def _prepare_agents_args(self):
        logger.info("Load configs")
        memory_config = load_json(
            os.path.join(self.scene_path, CONFIG_DIR, MEMORY_CONFIG)
        )
        model_configs = load_json(
            os.path.join(self.scene_path, CONFIG_DIR, MODEL_CONFIG)
        )
        agent_configs = [
            load_json(os.path.join(self.scene_path, CONFIG_DIR, config_path))
            for config_path in self.config["agent_configs_paths"].values()
        ]

        logger.info("Prepare agents args")
        llm_num = len(model_configs)
        agent_num = sum([len(config) for config in agent_configs])
        agent_num_per_llm = math.ceil(agent_num / llm_num)
        embedding_api_num = len(self.config["embedding_api"])
        logger.info(f"llm_num: {llm_num}")
        logger.info(f"agent_num: {agent_num}")
        logger.info(f"agent_num_per_llm: {agent_num_per_llm}")
        logger.info(f"embedding_api_num: {embedding_api_num}")
        memory_config["args"]["embedding_size"] = get_embedding_dimension(
            self.config["embedding_api"][0]
        )

        # Prepare agent args
        logger.info("Prepare agent args")
        index_ls = list(range(agent_num))
        random.shuffle(index_ls)
        for config, shuffled_idx in zip(
            list(itertools.chain.from_iterable(agent_configs)), index_ls
        ):
            model_config = model_configs[shuffled_idx // agent_num_per_llm]
            config["args"]["model_config_name"] = model_config["config_name"]
            config["args"]["memory_config"] = None if self.resume else memory_config
            config["args"]["embedding_api"] = self.config["embedding_api"][
                shuffled_idx % embedding_api_num
            ]

        return agent_configs if len(agent_configs) > 1 else agent_configs[0]

    def _set_env4agents(self):
        logger.info("Set all_agents for envs")
        agent_dict = {agent.agent_id: agent for agent in self.agents}
        results = []
        for env in self.envs:
            results.append(env.set_attr(attr="all_agents", value=agent_dict))
        for res in tqdm(results, total=len(self.envs), desc="Set all_agents for envs"):
            res.result()

    def _init_agents_envs(self, resume=False):
        raise NotImplementedError

    def _one_round(self):
        results = []
        for agent in self.agents:
            results.append(agent.run())
        outputs = []
        for res in results:
            output = res.result()
            outputs.append(output)
            logger.info(output)
        return outputs

    def run(self):
        raise NotImplementedError

    def load(self, file_path):
        with open(file_path, "rb") as f:
            return dill.load(f)

    def get_save_state(self):
        results = []
        if hasattr(self, "agents"):
            for agent in self.agents:
                results.append(agent.save())
            agent_save_state = []
            for res in tqdm(results, total=len(results), desc="Get agent save state"):
                agent_save_state.append(res.result())

        if hasattr(self, "envs"):
            results = []
            for env in self.envs:
                results.append(env.save())
            env_save_state = []
            for res in tqdm(results, total=len(results), desc="Get env save state"):
                env_save_state.append(res.result())

        return agent_save_state, env_save_state

    def save(self):
        try:
            file_manager = FileManager.get_instance()
            save_path = os.path.join(
                file_manager.run_dir, f"ROUND-{self.cur_round}.pkl"
            )
            self.agent_save_state, self.env_save_state = self.get_save_state()
            self.cur_round += 1
            with open(save_path, "wb") as f:
                dill.dump(self, f)
            logger.info(f"Saved simulator to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save simulator: {e}")
