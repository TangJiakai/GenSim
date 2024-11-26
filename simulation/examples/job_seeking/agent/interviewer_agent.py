import os
from typing import List
from jinja2 import Environment, FileSystemLoader

from agentscope.rpc import async_func

from simulation.helpers.utils import *
from simulation.helpers.constants import *
from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.base_env import BaseEnv


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = FileSystemLoader(os.path.join(scene_path, "prompts"))
env = Environment(loader=file_loader)
Template = env.get_template("interviewer_prompts.j2").module


class Job(dict):
    def __init__(self, 
                name: str, 
                jd: str, 
                jr: List[str], 
                company: str,
                salary: str,
                benefits: List[str],
                location: str,):
        super().__init__(name=name, jd=jd, jr=jr, company=company, salary=salary, benefits=benefits, location=location)
        self.name = name
        self.jd = jd
        self.jr = jr
        self.company = company
        self.salary = salary
        self.benefits = benefits
        self.location = location

    def __str__(self):
        jr_string = "\n".join([f"- {r}" for r in self.jr])
        benefits_string = "\n".join([f"- {b}" for b in self.benefits])
        return (
            f"### Position Name \n{self.name}\n"
            f"### Job Description \n{self.jd}\n"
            f"### Job Requirements \n{jr_string}\n"
            f"### Company \n{self.company}\n"
            f"### Salary: \n{self.salary}\n"
            f"### Benefits:\n{benefits_string}\n"
            f"### Location: \n{self.location}\n"
        )


class InterviewerAgent(BaseAgent):
    """Interviewer agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        jd: str,
        jr: list,
        company: str,
        salary: str,
        benefits: List[str],
        location: str,
        env: BaseEnv,
        embedding_api: str = None,
        embedding: list = None,
        memory_config: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.model_config_name = model_config_name
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        self.memory = None
        if memory_config:
            self.memory = setup_memory(memory_config)
            self.memory.model = self.model
            self.memory.embedding_api = embedding_api
            self.memory.get_tokennum_func = self.get_tokennum_func
        self.job = Job(name=name, jd=jd, jr=jr, company=company, salary=salary, benefits=benefits, location=location)
        self.embedding = embedding
        self.env = env

        self.update_profile()

    def update_profile(self):
        self._profile = self.job.__str__()

    def get_attr(self, attr):
        if attr == "job":
            job = {
                "Position Name": self.job['name'],
                "Job Description": self.job['jd'],
                "Job Requirements": self.job['jr'],
                "Company": self.job['company'],
                "Salary": self.job['salary'],
                "Benefits": self.job['benefits'],
                "Location": self.job['location'],
            }
            return job
        return super().get_attr(attr)

    def screening_cv(self, seeker_info: str):
        msg = get_assistant_msg()
        msg.instruction = Template.screening_cv_instruction()
        guided_choice = ["no", "yes"]
        msg.observation = Template.screening_cv_observation(seeker_info, guided_choice)
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]
        return response

    def interview(self, dialog: str):
        instruction = Template.interview_closing_instruction()
        guided_choice = ["no", "yes"]
        observation = Template.make_interview_decision_observation(dialog, guided_choice)
        msg = get_assistant_msg()
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]
        return response

    def receive_notification(self, seeker_name: str, is_accept: bool, **kwargs):
        self.observe(
            get_assistant_msg(
                Template.receive_notification_observation(seeker_name, is_accept)
            )
        )
        return "success"

    @async_func
    def run(self, **kwargs):
        return "Done"