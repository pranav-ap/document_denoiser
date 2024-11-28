import torch
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageDraw

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

denormalize = T.Compose([
    T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
])


def tensor_to_pil_image(tensor):
    tensor = tensor.clone().detach()

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    tensor = tensor.clone().detach()
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    tensor = (tensor * 255).byte()

    return T.ToPILImage()(tensor)


def create_side_by_side_image(tensor1, tensor2, padding=10):
    image1 = tensor_to_pil_image(tensor1)
    image2 = tensor_to_pil_image(tensor2)

    new_width = image1.width + image2.width + padding
    new_height = max(image1.height, image2.height)
    side_by_side_image = Image.new('RGB', (new_width, new_height))

    side_by_side_image.paste(image1, (0, 0))  
    side_by_side_image.paste(image2, (image1.width + padding, 0))

    return side_by_side_image


def create_three_image_row(tensor1, tensor2, tensor3, padding=10):
    tensor1 = denormalize(tensor1)
    image1 = tensor_to_pil_image(tensor1)
    tensor2 = denormalize(tensor2)
    image2 = tensor_to_pil_image(tensor2)
    tensor3 = denormalize(tensor3)
    image3 = tensor_to_pil_image(tensor3)

    new_width = image1.width + image2.width + image3.width + 2 * padding
    new_height = max(image1.height, image2.height, image3.height)
    row_image = Image.new('RGB', (new_width, new_height))

    row_image.paste(image1, (0, 0))
    row_image.paste(image2, (image1.width + padding, 0))
    row_image.paste(image3, (image1.width + image2.width + 2 * padding, 0))

    return row_image


def create_image_column(rows, padding=10):
    column_width = max(row.width for row in rows)
    column_height = sum(row.height for row in rows) + padding * (len(rows) - 1)
    column_image = Image.new('RGB', (column_width, column_height))

    y_offset = 0

    for row in rows:
        column_image.paste(row, (0, y_offset))
        y_offset += row.height + padding

    return column_image


def get_tensor_grid(pil_image):
    return to_tensor(pil_image).unsqueeze(0)
