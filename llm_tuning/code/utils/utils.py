from utils.constants import *
import os


def check_load_adapter(llm_path):
    if os.path.isfile(os.path.join(llm_path, ADAPTER_CONFIG_FILE_NAME)):
        return True
    return False


def check_dirs(llm_path):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory {SAVE_DIR}")

    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"Please download the LLM model to {llm_path}")

        