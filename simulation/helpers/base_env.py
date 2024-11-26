from typing import Dict, List
import dill

from agentscope.environment import BasicEnv
from agentscope.rpc import async_func

from simulation.helpers.utils import *


class BaseEnv(BasicEnv):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name)
        self.all_agents: Dict = dict()

    @async_func
    def load(self, data, **kwargs):
        state = dill.loads(data)
        self.__setstate__(state)
        return "success"

    @async_func
    def save(self):
        state = dill.dumps(self.__getstate__())
        return state

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @async_func
    def set_attr(self, attr: str, value, **kwargs) -> str:
        attrs = attr.split(".")
        obj = self
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
        return "success"

    def get_agents_by_ids(self, agent_ids: List[str]):
        agents = {agent_id: self.all_agents[agent_id] for agent_id in agent_ids}
        return agents

    def broadcast(self, content: str) -> None:
        # TODO: currently wait for each agent to finish processing the message (slow & unreasonable)
        results = []
        for agent in self.all_agents.values():
            results.append(agent.set_attr("global_intervention", content))
        for reuslt in results:
            reuslt.result()

    def intervention(self, agent_id: str, key, value) -> None:
        if agent_id in self.all_agents:
            agent = self.all_agents[agent_id]
            agent.set_attr(key, value).result()

    def interview(self, agent_id: str, query: str) -> str:
        if agent_id in self.all_agents:
            agent = self.all_agents[agent_id]
            return agent.external_interview(query)
