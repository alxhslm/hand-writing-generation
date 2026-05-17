import numpy as np
from PIL import Image, ImageOps


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
