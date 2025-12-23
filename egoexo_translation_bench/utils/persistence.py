"""Persistence utilities for saving and loading configuration and result data.

This module provides functions for handling JSON and YAML file I/O operations
used throughout the benchmark system for configuration management and result
persistence.
"""

import json
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

from . import utest


def save_results(results: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save results dictionary to a JSON file.
    
    Args:
        results: Dictionary containing results data to save
        path: File path where to save the JSON file
        
    Raises:
        IOError: If file cannot be written
        TypeError: If results cannot be JSON serialized
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(json_path: Union[str, Path] = 'results.json') -> Dict[str, Any]:
    """Load results from a JSON file.
    
    If the file is not found, returns dummy data for demonstration purposes.
    
    Args:
        json_path: Path to the JSON file to load
        
    Returns:
        Dictionary containing the loaded results data or dummy data if file not found
        
    Raises:
        json.JSONDecodeError: If the JSON file is malformed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.info(f"Error: {json_path} not found. Using dummy data for demonstration.")
        return utest.generate_dummy_data()


def load_yaml_config(yaml_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the loaded configuration, or None if file not found
        
    Raises:
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)  # Use safe_load for security
        return cfg
    except FileNotFoundError:
        logger.info(f"Error: {yaml_path} not found.")
        return None


def load_json_config(json_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load configuration from a JSON file.
    
    Args:
        json_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing the loaded configuration, or None if file not found
        
    Raises:
        json.JSONDecodeError: If the JSON file is malformed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.info(f"Error: {json_path} not found.")
        return None
