import json
import yaml
import logging
logger = logging.getLogger(__name__)

from . import utest

def save_results(results: dict, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    f.close()
    
def load_results(json_path='results.json'):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.info(f"Error: {json_path} not found. Using dummy data for demonstration.")
        return utest.generate_dummy_data()

def load_yaml_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            cfg = yaml.load(f)
        return cfg
    except FileNotFoundError:
        logger.info(f"Error: {yaml_path} not found.")

def load_json_config(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.info(f"Error: {json_path} not found.")
