import numpy as np
import pytest
import torch
from PIL import Image

from training.model import VAE
from training.utils import class_num_to_label, clip_image, generate_images, label_to_class_num, recenter_image

BATCH = 3
NUM_LATENT = 64
OUTPUT_SIZE = 36
INPUT_SIZE = 28


def make_model() -> VAE:
    return VAE(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_filters=32, num_latent_var=NUM_LATENT)


def test_class_num_to_label_digits() -> None:
    assert [class_num_to_label(i) for i in range(10)] == [str(i) for i in range(10)]


def test_class_num_to_label_letters() -> None:
    assert class_num_to_label(10) == "A"
    assert class_num_to_label(35) == "Z"


def test_label_to_class_num_digits() -> None:
    assert [label_to_class_num(str(i)) for i in range(10)] == list(range(10))


def test_label_to_class_num_letters() -> None:
    assert label_to_class_num("A") == 10
    assert label_to_class_num("Z") == 35


def test_label_to_class_num_invalid() -> None:
    assert label_to_class_num("a") is None
    assert label_to_class_num("!") is None
    assert label_to_class_num(" ") is None


@pytest.mark.parametrize("i", range(10))
def test_label_roundtrip(i: int) -> None:
    label = class_num_to_label(i)
    assert label_to_class_num(label) == i


@pytest.mark.parametrize("i", range(10, 36))
def test_label_roundtrip_letters(i: int) -> None:
    label = class_num_to_label(i)
    assert label_to_class_num(label) == i


def test_clip_image_all_zeros() -> None:
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    result = clip_image(img)
    assert np.array_equal(np.array(result), np.array(img))


def test_clip_image_with_content() -> None:
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[5:10, 8:15] = 255
    img = Image.fromarray(arr)
    result = clip_image(img)
    assert result.size == (7, 5)  # width=15-8, height=10-5


@pytest.mark.parametrize("randomness", [0.0, 0.5, 1.0])
def test_generate_images_output_shape(randomness: float) -> None:
    model = make_model()
    y = torch.tensor([0, 1, 2])
    z_means = torch.randn(BATCH, NUM_LATENT)
    log_z_vars = torch.randn(BATCH, NUM_LATENT)
    images = generate_images(model, y, z_means, log_z_vars, randomness)
    assert len(images) == len(y)
    assert all(im.shape == (INPUT_SIZE, INPUT_SIZE) for im in images)


def test_generate_images_values_in_range() -> None:
    model = make_model()
    y = torch.tensor([0])
    z_means = torch.randn(BATCH, NUM_LATENT)
    log_z_vars = torch.randn(BATCH, NUM_LATENT)
    images = generate_images(model, y, z_means, log_z_vars, 1.0)
    assert all(im.min() >= 0.0 and im.max() <= 1.0 for im in images)


def test_recenter_image_output_size() -> None:
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[5:15, 5:15] = 200
    img = Image.fromarray(arr)
    result = recenter_image(img)
    assert result.size == (28, 28)
