# -*- coding: utf-8 -*-
from datetime import datetime
import re
import os
import threading
from typing import Optional, Sequence, Union
from jinja2 import Environment, FileSystemLoader
from typing import Optional
import math
import faiss
import numpy as np

from agentscope.models import ModelResponse
from agentscope.message import Msg

import simulation.memory.short_memory as ShortMemory
from simulation.helpers.emb_service import *


file_loader = FileSystemLoader(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
)
env = Environment(loader=file_loader)
Template = env.get_template("prompts.j2").module


class ShortLongMemory(ShortMemory.ShortMemory):
    def __init__(
        self,
        *,
        embedding_size: int,
        importance_weight: Optional[float] = 0.15,
        stm_K: int = 2,
        ltm_K: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(stm_K=stm_K, **kwargs)
        self.importance_weight = importance_weight
        self.ltm_K = ltm_K

        self.ltm_memory = []
        self.retriever = faiss.IndexFlatL2(embedding_size)

        self.model, self.embedding_api = None, None
        self.get_tokennum_func = None

        self.ltm_lock = threading.Lock()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("ltm_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.ltm_lock = threading.Lock()

    def _score_memory_importance(self, memory_content: str) -> float:
        msg = Msg(
            "assistant",
            Template.score_importance_prompt(memory_content),
            role="assistant",
        )
        prompt = self.model.format(msg)

        def parse_func(response: ModelResponse) -> ModelResponse:
            try:
                match = re.search(r"^\D*(\d+)", response.text.strip())
                if match:
                    res = (float(match.group(1)) / 10) * self.importance_weight
                else:
                    res = 0.0
                return ModelResponse(raw=res)
            except:
                raise ValueError(
                    f"Invalid response format in parse_func "
                    f"with response: {response.text}",
                )

        response = self.model(prompt, parse_func=parse_func).raw
        return response

    def add_ltm_memory(self, ltm_memory_unit: Msg):
        memory_content = ltm_memory_unit.content
        ltm_memory_unit.importance_score = self._score_memory_importance(memory_content)

        with self.ltm_lock:
            self.ltm_memory.append(ltm_memory_unit)
            self.retriever.add(
                np.atleast_2d(get_embedding(memory_content, self.embedding_api))
            )

    def get_salient_docs(self, query: Msg, k=3):
        def relevance_score_fn(score: float) -> float:
            """Return a similarity score on a scale [0, 1]."""
            # This will differ depending on a few things:
            # - the distance / similarity metric used by the VectorStore
            # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
            # This function converts the euclidean norm of normalized embeddings
            # (0 is most similar, sqrt(2) most dissimilar)
            # to a similarity function (0 to 1)
            return 1.0 - score / math.sqrt(2)

        scores, indices = self.retriever.search(np.atleast_2d(query.embedding), k)
        docs_and_scores = {}
        for j, i in enumerate(indices[0]):
            if i == -1:
                continue
            docs_and_scores[i] = (self.ltm_memory[i], relevance_score_fn(scores[0][j]))
        return docs_and_scores

    def _get_combined_score(self, query, doc, relevance_score):
        def score_func(m1: Msg, m2: Msg) -> float:
            time_gap = (
                datetime.strptime(m1.timestamp, "%Y-%m-%d %H:%M:%S")
                - datetime.strptime(m2.timestamp, "%Y-%m-%d %H:%M:%S")
            ).total_seconds() / 3600
            recency = 0.99**time_gap
            return recency

        score = score_func(query, doc)
        score += relevance_score
        score += doc.importance_score

        return score

    def _get_rescored_docs(self, query, docs_and_scores):
        rescored_docs = [
            (doc, self._get_combined_score(query, doc, relevance_score))
            for doc, relevance_score in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in rescored_docs[: self.ltm_K]]

    def add(self, memory: Union[Sequence[Msg], Msg, None] = None):
        if memory is None:
            return None

        ltm_memory_unit = super().add(memory)
        if ltm_memory_unit:
            self.add_ltm_memory(ltm_memory_unit)

    def get_ltm_memory(self, query: Msg):
        query.embedding = get_embedding(query.content, self.embedding_api)
        docs_and_scores = {
            len(self.ltm_memory) - self.ltm_K + i: (doc, 0.0)
            for i, doc in enumerate(self.ltm_memory[-self.ltm_K :])
        }

        docs_and_scores.update(self.get_salient_docs(query))
        return self._get_rescored_docs(query, docs_and_scores)

    def get_memory(self, query: Msg = None, only_short=False):
        stm_memory = super().get_memory(query)
        if not query or only_short:
            return stm_memory
        return self.get_ltm_memory(query) + stm_memory
