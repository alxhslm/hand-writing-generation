import numpy as np
import torch
from PIL import Image, ImageOps

from training.model import VAE


def generate_images(
    model: VAE, y: torch.Tensor, z_means: torch.Tensor, log_z_vars: torch.Tensor, randomness: float
) -> list[np.ndarray]:
    z_evals = model.sample(z_means, log_z_vars + 2 * np.log(randomness + 1e-8)).tile(len(y), 1)
    y_evals = (
        torch.nn.functional.one_hot(y.reshape(-1, 1).tile(1, z_means.shape[0]).flatten(), num_classes=model.output_size)
        .float()
    )
    return [
        im
        for im in model.decode(z_evals, y_evals)
        .sigmoid()
        .reshape(len(y), z_means.shape[0], model.input_size, model.input_size)
        .mean(dim=1)
        .detach()
        .numpy()
    ]


def class_num_to_label(y: int) -> str:
    if y < 10:
        return str(y)
    return chr(y - 10 + ord("A"))


def label_to_class_num(label: str) -> int | None:
    if label.isnumeric():
        return int(label)
    if label < "A" or label > "Z":
        return None
    return ord(label) - ord("A") + 10


def clip_image(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    if (img_array == 0).all():
        return image
    img_array = img_array[np.any(img_array > 1e-3, 1), :]
    img_array = img_array[:, np.any(img_array > 1e-3, 0)]
    return Image.fromarray(img_array)


def recenter_image(image: Image.Image) -> Image.Image:
    image = clip_image(image)
    image = ImageOps.pad(image, (20, 20))
    return ImageOps.expand(image, (4, 4))
