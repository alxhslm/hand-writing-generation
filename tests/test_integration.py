import numpy as np
import pytest
import torch

from training.model import VAE
from training.utils import generate_images

MODEL_PATH = "model.pt"
INPUT_SIZE = 28
OUTPUT_SIZE = 36
NUM_LATENT = 64


@pytest.fixture(scope="module")
def model() -> VAE:
    m = VAE(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_filters=32, num_latent_var=NUM_LATENT)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    m.eval()
    return m


def test_model_loads(model: VAE) -> None:
    assert isinstance(model, VAE)


def test_encode_with_real_weights(model: VAE) -> None:
    x = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
    y_pred, z_mean, log_z_var = model.encode(x)
    assert y_pred.shape == (1, OUTPUT_SIZE)
    assert z_mean.shape == (1, NUM_LATENT)
    assert log_z_var.shape == (1, NUM_LATENT)


def test_generate_images_with_real_weights(model: VAE) -> None:
    x = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
    _, z_mean, log_z_var = model.encode(x)
    images = generate_images(model, torch.tensor([0]), z_mean, log_z_var, randomness=1.0)
    assert len(images) == 1
    im = images[0]
    assert im.shape == (INPUT_SIZE, INPUT_SIZE)
    assert im.min() >= 0.0 and im.max() <= 1.0


def test_full_pipeline(model: VAE) -> None:
    batch = torch.zeros(5, 1, INPUT_SIZE, INPUT_SIZE)
    _, z_mean, log_z_var = model.encode(batch)
    for class_idx in range(OUTPUT_SIZE):
        images = generate_images(model, torch.tensor([class_idx]), z_mean, log_z_var, randomness=1.0)
        assert len(images) == 1
        assert images[0].shape == (INPUT_SIZE, INPUT_SIZE)
        assert not np.isnan(images[0]).any()
