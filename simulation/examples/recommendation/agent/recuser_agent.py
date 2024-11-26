import random
import os
import jinja2
from loguru import logger

from agentscope.rpc import async_func

from simulation.examples.recommendation.environment.env import RecommendationEnv
from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.utils import (
    setup_memory,
    get_assistant_msg,
)
from simulation.helpers.constants import *


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("recuser_prompts.j2").module


class RecUserAgent(BaseAgent):
    """recuser agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        profile: str,
        env: RecommendationEnv,
        embedding_api: str = None,
        memory_config: dict = None,
        relationship: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        if memory_config is not None:
            self.memory = setup_memory(memory_config)
            self.memory.embedding_api = embedding_api
            self.memory.model = self.model
            self.memory.get_tokennum_func = self.get_tokennum_func
        self.env = env
        self._profile = f"- Name: {self.name} - Profile: {profile}"
        self.relationship = relationship

        self._state = "idle"

    def __getstate__(self) -> object:
        state = super().__getstate__()
        state.pop("relationship", None)
        return state

    def generate_feeling(self, movie):
        instruction = Template.generate_feeling_instruction()
        observation = Template.generate_feeling_observation(movie)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        feeling = self(msg).content

        logger.info(f"[{self.name}] feels {feeling}")
        return feeling

    def rating_item(self, movie):
        instruction = Template.rating_item_instruction()
        guided_choice = [
            "Rating 0.5",
            "Rating 1.0",
            "Rating 1.5",
            "Rating 2.0",
            "Rating 2.5",
            "Rating 3.0",
            "Rating 3.5",
            "Rating 4.0",
            "Rating 4.5",
            "Rating 5.0",
        ]
        observation = Template.rating_item_observation(movie, guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]
        action = response.split(":")[0]

        logger.info(f"[{self.name}] rated {action} for movie {movie}")

        return action

    def recommend(self):
        user_info = (
            self.profile
            + "\nMemory:"
            + "\n- ".join([m.content for m in self.memory.get_memory()])
        )
        guided_choice = self.env.recommend4user(user_info)
        instruction = Template.recommend_instruction()
        observation = Template.make_choice_observation(guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]["title"]

        logger.info(f"[{self.name}] selected movie {response}")

        feeling = self.generate_feeling(response)
        rating = self.rating_item(response)

    def conversation(self):
        friend_agent_id = random.choice(list(self.relationship.keys()))
        friend_agent = self.relationship[friend_agent_id]
        announcement = Template.conversation_instruction()
        dialog_observation = self.chat(announcement, [self, friend_agent])

        self.observe(get_assistant_msg(announcement + dialog_observation))
        friend_agent.observe(get_assistant_msg(announcement + dialog_observation))

        logger.info(
            f"[{self.name}] had a conversation with {friend_agent_id}: {dialog_observation}"
        )

        return dialog_observation

    def post(self):
        instruction = Template.post_instruction()
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = "Please give your post content."
        response = self(msg).content

        for agent in self.relationship.values():
            agent.observe(get_assistant_msg(f"{self.name} posted: {response}"))

        logger.info(f"[{self.name}] posted: {response}")

        return response

    @async_func
    def run(self, **kwargs):
        instruction = Template.start_action_instruction()
        guided_choice = [
            "Recommend: Request the website to recommend a batch of movies to watch.",
            "Conversation: Start a conversation with a good friend about a movie you've recently heard about or watched.",
            "Post: Post in your social circle expressing your recent thoughts on movie-related topics.",
        ]
        observation = Template.make_choice_observation(guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        answer = self.reply(msg).content
        response = guided_choice[int(answer)]
        logger.info(f"[{self.name}] selected action: {response}")
        action = response.split(":")[0].strip().lower()
        getattr(self, action)()

        logger.info("Finished running recuser agent.")

        return "Done"
