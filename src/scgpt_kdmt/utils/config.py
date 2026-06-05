"""Configuration loading utilities."""

import yaml
import os


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to a .yaml configuration file.

    Returns:
        Dictionary of configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("data_path", "output_dir", "checkpoint_path", "teacher_checkpoint"):
        if key in config and isinstance(config[key], str):
            if not os.path.isabs(config[key]):
                config[key] = os.path.normpath(
                    os.path.join(config_dir, config[key])
                )

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Merge override config into base config (shallow merge for top-level keys)."""
    merged = base.copy()
    merged.update(override)
    return merged