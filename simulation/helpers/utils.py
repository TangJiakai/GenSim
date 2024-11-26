import time
import requests
import yaml
import os
import json
import os
from loguru import logger

from agentscope.constants import _DEFAULT_CFG_NAME
from agentscope.manager import FileManager
from agentscope.message import Msg
from simulation.memory import *


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_configs(configs):
    file_manager = FileManager.get_instance()
    with open(
        os.path.join(file_manager.run_dir, _DEFAULT_CFG_NAME),
        "r+",
        encoding="utf-8",
    ) as file:
        cfg = json.load(file)
        cfg.update(configs)
        file.seek(0)
        json.dump(cfg, file, indent=4)
        file.truncate()


def setup_memory(memory_config):

    memory = eval(memory_config["class"])(**memory_config["args"])
    return memory


def get_assistant_msg(content=None, **kwargs):
    return Msg("assistant", content, role="assistant", **kwargs)


def get_memory_until_limit(memory, get_tokennum_func, existing_prompt=None, limit=6000):
    """
    Get memory until the total length of memory is less than limit
    """
    limited_memory = []
    if existing_prompt:
        limit -= get_tokennum_func(existing_prompt)
    for m in memory:
        if num_token := get_tokennum_func(m.content) < limit:
            limited_memory.append(m)
            limit -= num_token
        else:
            break
    return limited_memory


def get_token_num(prompt, url, model, api_key, backoff_factor=1):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": model, "prompt": prompt}

    attempt = 0
    while True:
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(data), timeout=60000
            )

            if response.status_code == 200:
                response_data = response.json()
                token_count = response_data.get("count", 0)
                return token_count
            else:
                logger.error(f"Error: {response.status_code}, {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")

        attempt += 1
        sleep_time = backoff_factor * (2**attempt)
        logger.info(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
