import faiss
import numpy as np

from simulation.helpers.utils import *
from simulation.helpers.emb_service import *
from simulation.helpers.base_env import BaseEnv


class RecommendationEnv(BaseEnv):
    def __init__(
        self,
        name: str,
        embedding_api: str,
        item_infos: list,
        index: faiss.Index,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.item_infos = item_infos
        self.embedding_api = embedding_api
        self.index = faiss.deserialize_index(index) if index is not None else None
        self.all_agents = None

    def __getstate__(self) -> object:
        state = super().__getstate__()
        state["index"] = faiss.serialize_index(self.index)
        return state

    def __setstate__(self, state) -> None:
        if "index" in state:
            state["index"] = faiss.deserialize_index(state["index"])
        super().__setstate__(state)

    def recommend4user(self, user_info, k=5):
        user_emb = get_embedding(user_info, self.embedding_api)
        _, indices = self.index.search(np.array([user_emb]), k)
        return [self.item_infos[i] for i in indices[0]]
