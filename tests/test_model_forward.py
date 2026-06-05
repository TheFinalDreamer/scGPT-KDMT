"""Test model forward passes."""

import pytest
import torch
from src.scgpt_kdmt.models.student import StudentModel
from src.scgpt_kdmt.models.transformer import TransformerEncoder
from src.scgpt_kdmt.models.classifier import Classifier


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_transformer_forward(device):
    model = TransformerEncoder(input_dim=2000, hidden_dim=128, num_heads=4, num_layers=3).to(device)
    x = torch.randn(4, 2000).to(device)
    out = model(x)
    assert out.shape == (4, 128)


def test_classifier_forward(device):
    model = Classifier(input_dim=128, num_classes=31).to(device)
    x = torch.randn(4, 128).to(device)
    out = model(x)
    assert out.shape == (4, 31)


def test_student_forward(device):
    model = StudentModel(
        input_dim=2000, hidden_dim=128, num_heads=4, num_layers=3, num_classes=31
    ).to(device)
    x = torch.randn(4, 2000).to(device)
    outputs = model(x)
    assert "embedding" in outputs
    assert "logits" in outputs
    assert "recon" in outputs
    assert "masked_pred" in outputs
    assert outputs["embedding"].shape == (4, 128)
    assert outputs["logits"].shape == (4, 31)
    assert outputs["recon"].shape == (4, 2000)
    assert outputs["masked_pred"].shape == (4, 2000)


def test_student_count_parameters():
    model = StudentModel(
        input_dim=2000, hidden_dim=128, num_heads=4, num_layers=3, num_classes=31
    )
    n_params = sum(p.numel() for p in model.parameters())
    # Should be approximately 2.92M
    assert 2.5e6 < n_params < 3.5e6