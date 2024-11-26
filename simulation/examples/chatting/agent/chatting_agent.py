# -*- coding: utf-8 -*-
"""An env used as a chatroom."""
from typing import Any, Union, Generator, Tuple
import threading
import os
import jinja2
from loguru import logger

from agentscope.rpc import async_func

from agentscope.message import Msg
from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.utils import setup_memory
from simulation.examples.chatting.environment.env import ChatRoom


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("chatting_prompts.j2").module


class ChatRoomAgent(BaseAgent):
    """A agent with chat room"""

    def __init__(  # pylint: disable=W0613
        self,
        name: str,
        model_config_name: str,
        profile: str,
        env: ChatRoom,
        embedding_api: str = None,
        memory_config: dict = None,
        **kwargs: Any,
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
        self._profile = f"### Name: {self.name}\n" f"### Profile: {profile}"

        self.room = None
        self.mentioned_messages = []
        self.mentioned_messages_lock = threading.Lock()

    def add_mentioned_message(self, msg: Msg) -> None:
        """Add mentioned messages"""
        with self.mentioned_messages_lock:
            self.mentioned_messages.append(msg)

    def join(self, room: ChatRoom) -> bool:
        """Join a room"""
        self.room = room
        return room.join(self)

    def generate_hint(self) -> Msg:
        """Generate a hint for the agent"""
        if self.mentioned_messages:
            hint = (
                self.profile
                + r"""\n\nYou have be mentioned in the following message, """
                r"""please generate an appropriate response."""
            )
            for message in self.mentioned_messages:
                hint += f"\n{message.name}: " + message.content
            self.mentioned_messages = []
            return Msg("system", hint, role="system")
        else:
            return Msg("system", self.profile, role="system")

    def speak(
        self,
        content: Union[str, Msg, Generator[Tuple[bool, str], None, None]],
    ) -> None:
        """Speak to room.

        Args:
            content
            (`Union[str, Msg, Generator[Tuple[bool, str], None, None]]`):
                The content of the message to be spoken in chatroom.
        """
        super().speak(content)
        self.room.speak(content)

    def reply(self, x: Msg = None) -> Msg:
        """Generate reply to chat room"""
        msg_hint = self.generate_hint()
        self_msg = Msg(name=self.name, content="", role="assistant")

        history = self.room.get_history(self.agent_id)
        prompt = self.model.format(
            msg_hint,
            history,
            self_msg,
        )
        logger.debug(prompt)
        response = self.model(
            prompt,
            parse_func=self.room.chatting_parse_func,
        ).text
        msg = Msg(name=self.name, content=response, role="assistant")
        if response:
            self.speak(msg)
        return msg

    @async_func
    def run(self, **kwargs):
        self.join(self.env)
        return "Done"
