import random
import os
import jinja2
from loguru import logger

from agentscope.message import Msg
from agentscope.rpc import async_func

from simulation.helpers.base_agent import BaseAgent
from simulation.helpers.utils import *
from simulation.helpers.constants import *
from simulation.helpers.base_env import BaseEnv


scene_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_loader = jinja2.FileSystemLoader(os.path.join(scene_path, "prompts"))
env = jinja2.Environment(loader=file_loader)
Template = env.get_template("seeker_prompts.j2").module


class Seeker(object):
    def __init__(self, name: str, cv: str, trait: str):
        self.name = name
        self.cv = cv
        self.trait = trait
        self.working_condition = "unemployed"

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"CV: {self.cv}\n"
            f"Current Working Condition: {self.working_condition}\n"
        )


class SeekerAgent(BaseAgent):
    """seeker agent."""

    def __init__(
        self,
        name: str,
        model_config_name: str,
        cv: str,
        trait: str,
        env: BaseEnv,
        embedding: list = None,
        embedding_api: str = None,
        memory_config: dict = None,
        job_ids_pool: list[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            model_config_name=model_config_name,
        )
        self.memory_config = memory_config
        self.embedding_api = embedding_api
        self.memory = None
        if memory_config:
            self.memory = setup_memory(memory_config)
            self.memory.embedding_api = embedding_api
            self.memory.model = self.model
            self.memory.get_tokennum_func = self.get_tokennum_func
        self.job_ids_pool = job_ids_pool
        self.embedding = embedding
        self.env = env

        self.seeker = Seeker(name, cv, trait)
        self._update_profile()

    def _update_profile(self):
        cv = self.seeker.cv
        cv_mdstr = ""
        level = 4

        for key, value in cv.items():
            cv_mdstr += f"{'#' * level} {key}\n"  # Add the section title
            level += 1  # Increase level for the next section

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):  # For work experience
                        cv_mdstr += (
                            f"- {'#' * (level + 1)} Company: {item['Company']}\n"
                        )
                        cv_mdstr += f"\tPosition: {item['Position']}\n"
                        cv_mdstr += f"\tTime: {item['Time']}\n"
                    else:  # For skills
                        cv_mdstr += f"- {item}\n"
            else:
                cv_mdstr += f"{value}\n"

            level -= 1  # Reset level for the next key

        cv_mdstr = cv_mdstr.strip()

        trait = self.seeker.trait
        trait_mdstr = "\n".join(
            [f"#### {key}\n{value}" for key, value in trait.items()]
        )

        self._profile = (
            f"### Name \n{self.seeker.name}\n"
            f"### CV \n{cv_mdstr}\n"
            f"### Trait \n{trait_mdstr}\n"
            f"### Working Condition \n{self.seeker.working_condition}"
        )

    def _determine_if_seeking(self, **kwargs):
        instruction = Template.determine_if_seeking_instruction()
        guided_choice = ["no", "yes"]
        observation = Template.make_choice_observation(guided_choice)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]
        return response

    def _determine_search_job_number(self, **kwargs):
        """Set search job number."""
        SearchJobNumber = 5

        instruction = Template.determine_search_job_number_instruction()
        guided_choice = list(map(str, range(1, SearchJobNumber + 1)))
        observation = Template.make_choice_observation(guided_choice)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]
        return int(response)

    def _determine_search_jobs(self, search_job_number: int, **kwargs):
        search_job_indices = random.sample(
            range(len(self.job_ids_pool)), search_job_number
        )
        search_job_ids = [self.job_ids_pool[i] for i in search_job_indices]
        interviewer_agent_infos = self.env.get_agents_by_ids(search_job_ids)

        for agent in interviewer_agent_infos.values():
            agent.job = agent.get_attr("job")

        return interviewer_agent_infos

    def _determine_apply_job(self, interviewer_agent_infos: dict, **kwargs):
        """Determine which jobs to apply."""
        instruction = Template.determine_apply_jobs_instruction()
        apply_interviewer_agent_infos = {}
        guided_choice = ["no", "yes"]
        for job_id, agent in interviewer_agent_infos.items():
            job_info = agent.job
            observation = Template.determine_apply_jobs_observation(
                job_info, guided_choice
            )
            msg = Msg("user", None, role="user")
            msg.instruction = instruction
            msg.observation = observation
            msg.guided_choice = list(map(str, range(len(guided_choice))))
            response = guided_choice[int(self.reply(msg).content)]

            if response == "yes":
                apply_interviewer_agent_infos[job_id] = agent

        return apply_interviewer_agent_infos

    def _apply_job(self, apply_interviewer_agent_infos: dict, **kwargs):
        """Apply jobs."""
        results = []
        cv_passed_interviewer_agent_infos = {}
        for agent_id, agent in apply_interviewer_agent_infos.items():
            results.append(agent.screening_cv(str(self.seeker)))

        for (agent_id, agent), result in zip(
            apply_interviewer_agent_infos.items(),
            results,
        ):
            if "yes" == result:
                cv_passed_interviewer_agent_infos[agent_id] = agent

        if len(cv_passed_interviewer_agent_infos) > 0:
            self.observe(
                get_assistant_msg(
                    Template.apply_job_observation(cv_passed_interviewer_agent_infos)
                )
            )

        return cv_passed_interviewer_agent_infos

    def _interview_fun(self, cv_passed_interviewer_agent_infos: dict, **kwargs):
        """Interview."""
        results = []
        offer_interviewer_agent_infos = {}
        for agent_id, agent in cv_passed_interviewer_agent_infos.items():
            announcement = Template.interview_announcement_instruction()
            dialog_observation = self.chat(announcement, [self, agent])
            self.observe(get_assistant_msg(announcement + dialog_observation))
            results.append(agent.interview(dialog_observation))

        for (agent_id, agent), result in zip(
            cv_passed_interviewer_agent_infos.items(), results
        ):
            if "yes" == result:
                offer_interviewer_agent_infos[agent_id] = agent
                self.observe(
                    get_assistant_msg(Template.interview_observation(agent.job, True))
                )
            else:
                self.observe(
                    get_assistant_msg(Template.interview_observation(agent.job, False))
                )

        return offer_interviewer_agent_infos

    def _make_final_decision(self, offer_interviewer_agent_infos: dict, **kwargs):
        """Make decision."""
        if len(offer_interviewer_agent_infos) == 0:
            return -1

        if len(offer_interviewer_agent_infos) == 1:
            agent = list(offer_interviewer_agent_infos.values())[0]
            final_job = agent.job
            self.seeker.working_condition = (
                "Position Name: " + final_job["Position Name"]
            )
            self._update_profile()
            agent.receive_notification(self.seeker.name, True)
            return list(offer_interviewer_agent_infos.keys())[0]

        instruction = Template.make_final_decision_instruction()
        jobs = {
            agent.agent_id: agent.job
            for agent in offer_interviewer_agent_infos.values()
        }
        guided_choice = list(offer_interviewer_agent_infos.keys())
        observation = Template.make_final_decision_observation(jobs, guided_choice)
        msg = Msg("user", None, role="user")
        msg.instruction = instruction
        msg.observation = observation
        msg.guided_choice = list(map(str, range(len(guided_choice))))
        response = guided_choice[int(self.reply(msg).content)]

        final_job = offer_interviewer_agent_infos[response].job
        self.seeker.working_condition = "Position Name: " + final_job["Position Name"]
        self._update_profile()

        for agent_id, agent in offer_interviewer_agent_infos.items():
            agent.receive_notification(self.seeker.name, agent_id == response)

        return response

    @async_func
    def run(self, **kwargs):
        if "no" in self._determine_if_seeking():
            logger.info("No seeking job.")
            return "No Seeking Job."

        search_job_number = self._determine_search_job_number()
        logger.info(f"Search job number: {search_job_number}")

        interviewer_agent_infos = self._determine_search_jobs(search_job_number)
        logger.info(f"Search jobs: {list(interviewer_agent_infos.keys())}")

        apply_interviewer_agent_infos = self._determine_apply_job(
            interviewer_agent_infos
        )
        logger.info(f"Apply jobs: {list(apply_interviewer_agent_infos.keys())}")

        cv_passed_interviewer_agent_infos = self._apply_job(
            apply_interviewer_agent_infos
        )
        logger.info(f"CV passed jobs: {list(cv_passed_interviewer_agent_infos.keys())}")

        offer_interviewer_agent_infos = self._interview_fun(
            cv_passed_interviewer_agent_infos
        )
        logger.info(f"Offer jobs: {list(offer_interviewer_agent_infos.keys())}")

        final_job_id = self._make_final_decision(offer_interviewer_agent_infos)
        logger.info(f"Final job: {final_job_id}")

        return final_job_id
