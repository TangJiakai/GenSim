from datetime import datetime
import random
import requests
import time
from loguru import logger


MAX_TIMEOUT_DELAY = 60
session = requests.Session()


def get_embedding(sentence, api, delay=5):
    url = f"{api}/encode"
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.post(url, json={"sentence": sentence}, timeout=60)
            response.raise_for_status()
            embedding = response.json().get("embedding")
            return embedding
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt} to get embedding failed. Retrying after {delay} seconds..."
            )
            delay = min(2 * delay, MAX_TIMEOUT_DELAY)
            delay = (random.random() + 0.5) * delay
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Request failed with error: {e}, Sentence: {sentence}, URL: {url}"
            )


def get_embedding_dimension(api, delay=5):
    url = f"{api}/embedding-dimension"
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            embedding_dimension = response.json().get("embedding_dimension")
            return embedding_dimension
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt} to get embedding failed. Retrying after {delay} seconds..."
            )
            delay = min(2 * delay, MAX_TIMEOUT_DELAY)
            delay = (random.random() + 0.5) * delay
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed with error: {e}, URL: {url}")
