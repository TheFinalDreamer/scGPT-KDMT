"""Test that all core modules can be imported."""

import pytest


def test_import_data():
    from src.scgpt_kdmt.data.dataset import SingleCellDataset, build_dataloader
    from src.scgpt_kdmt.data.split import split_dataset
    from src.scgpt_kdmt.data.masking import mask_input


def test_import_models():
    from src.scgpt_kdmt.models.student import StudentModel
    from src.scgpt_kdmt.models.transformer import TransformerEncoder
    from src.scgpt_kdmt.models.classifier import Classifier
    from src.scgpt_kdmt.models.masked_prediction import MaskedGenePredictor


def test_import_losses():
    from src.scgpt_kdmt.losses.kd_loss import kd_loss, feature_distillation_loss
    from src.scgpt_kdmt.losses.contrastive import supervised_contrastive_loss


def test_import_evaluation():
    from src.scgpt_kdmt.evaluation.metrics import (
        classification_metrics,
        compute_confusion_matrix,
        clustering_metrics,
    )


def test_import_utils():
    from src.scgpt_kdmt.utils.config import load_config
    from src.scgpt_kdmt.utils.seed import set_seed
    from src.scgpt_kdmt.utils.io import save_json, load_json