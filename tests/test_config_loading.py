"""Test configuration loading."""

import os
import pytest
from src.scgpt_kdmt.utils.config import load_config, merge_configs


def test_load_student_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "student_kdmt.yaml"
    )
    config = load_config(config_path)
    assert "input_dim" in config
    assert config["input_dim"] == 2000
    assert config["hidden_dim"] == 128
    assert "use_kd" in config


def test_load_teacher_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "teacher_scgpt.yaml"
    )
    config = load_config(config_path)
    assert "teacher_hidden_dim" in config
    assert config["teacher_hidden_dim"] == 512
    assert "num_epochs" in config


def test_load_ablation_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "ablation.yaml"
    )
    config = load_config(config_path)
    assert "input_dim" in config
    assert config["input_dim"] == 2000


def test_merge_configs():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    merged = merge_configs(base, override)
    assert merged["a"] == 1
    assert merged["b"] == 3
    assert merged["c"] == 4