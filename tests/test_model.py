import torch

from training.model import VAE

BATCH = 4
INPUT_SIZE = 28
OUTPUT_SIZE = 36
NUM_FILTERS = 32
NUM_LATENT = 64


def make_model() -> VAE:
    return VAE(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_filters=NUM_FILTERS, num_latent_var=NUM_LATENT)


def test_encode_shapes() -> None:
    model = make_model()
    x = torch.randn(BATCH, 1, INPUT_SIZE, INPUT_SIZE)
    y_pred, z_mean, log_z_var = model.encode(x)
    assert y_pred.shape == (BATCH, OUTPUT_SIZE)
    assert z_mean.shape == (BATCH, NUM_LATENT)
    assert log_z_var.shape == (BATCH, NUM_LATENT)


def test_decode_shape() -> None:
    model = make_model()
    z = torch.randn(BATCH, NUM_LATENT)
    y = torch.randn(BATCH, OUTPUT_SIZE).softmax(dim=-1)
    x_hat = model.decode(z, y)
    assert x_hat.shape == (BATCH, 1, INPUT_SIZE, INPUT_SIZE)


def test_forward_shapes() -> None:
    model = make_model()
    x = torch.randn(BATCH, 1, INPUT_SIZE, INPUT_SIZE)
    x_hat, z_mean, log_z_var, y_pred = model(x)
    assert x_hat.shape == (BATCH, 1, INPUT_SIZE, INPUT_SIZE)
    assert z_mean.shape == (BATCH, NUM_LATENT)
    assert log_z_var.shape == (BATCH, NUM_LATENT)
    assert y_pred.shape == (BATCH, OUTPUT_SIZE)


def test_sample_shape() -> None:
    model = make_model()
    z_mean = torch.randn(BATCH, NUM_LATENT)
    log_z_var = torch.randn(BATCH, NUM_LATENT)
    z = model.sample(z_mean, log_z_var)
    assert z.shape == (BATCH, NUM_LATENT)
