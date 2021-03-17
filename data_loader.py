from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as t
import glob
import torch
from torch import tensor
from PIL import Image

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as t
import glob
import torch
from torch import tensor
from PIL import Image
from typing import Tuple

data_path = '/content/drive/MyDrive/day_night_detection'


def path_to_sq_tensor(p: str) -> tensor:
    image = Image.open(p)
    image = resize_image(image, 360)
    tens = t.ToTensor()(image)
    return tens


def all_true(data: list) -> Tuple[bool, list]:
    res = True
    false_indexes = []
    for i, item in enumerate(data):
        if not item:
            res = False
            false_indexes.append(i)
    return res, false_indexes


class ImagesDataset(Dataset):
    def __init__(self, data):
        """
        :param data: either path to a root folder containing two sub folders: /day/, /night/ with corresponding images
                     or a list of mongo item
        """
        super(ImagesDataset, self).__init__()
        import os
        if type(data) == str:
            assert os.path.exists(data), f'path {data} does not exist'

            path_to_data = data
            day_images = glob.glob(f'{path_to_data}/**/day/**.jpg', recursive=True)
            night_images = glob.glob(f'{path_to_data}/**/night/**.jpg', recursive=True)

        else:
            assert type(data) == list, f'data should be either a list or a string'

            day_images = [p['path_image'] for p in data if 'day' in p['scene']]
            night_images = [p['path_image'] for p in data if 'day' not in p['scene']]

        labels = [0 for _ in range(len(day_images))]
        labels.extend([1 for _ in range(len(night_images))])

        data = day_images
        data.extend(night_images)

        self.data = data
        self.labels = labels

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        image_path = self.data[index]

        label = self.labels[index]

        ten = path_to_sq_tensor(image_path)

        return ten, tensor(label), image_path


def resize_image(image: Image, length: int) -> Image:
    """
    Resize an image to a square. Can make an image bigger to make it fit or smaller if it doesn't fit. It also crops
    part of the image.

    :param self:
    :param image: Image to resize.
    :param length: Width and height of the output image.
    :return: Return the resized image.
    """

    """
    Resizing strategy : 
     1) We resize the smallest side to the desired dimension (e.g. 1080)
     2) We crop the other side so as to make it fit with the same length as the smallest side (e.g. 1080)
    """
    if image.size[0] < image.size[1]:
        # The image is in portrait mode. Height is bigger than width.

        # This makes the width fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((length, int(image.size[1] * (length / image.size[0]))))

        # Amount of pixel to lose in total on the height of the image.
        required_loss = (resized_image.size[1] - length)

        # Crop the height of the image so as to keep the center part.
        resized_image = resized_image.crop(
            box=(0, required_loss / 2, length, resized_image.size[1] - required_loss / 2))

        # We now have a length*length pixels image.
        return resized_image
    else:
        # This image is in landscape mode or already squared. The width is bigger than the heihgt.

        # This makes the height fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((int(image.size[0] * (length / image.size[1])), length))

        # Amount of pixel to lose in total on the width of the image.
        required_loss = resized_image.size[0] - length

        # Crop the width of the image so as to keep 1080 pixels of the center part.
        resized_image = resized_image.crop(
            box=(required_loss / 2, 0, resized_image.size[0] - required_loss / 2, length))

        # We now have a length*length pixels image.
        return resized_image


if __name__ == '__main__':
    pass


    transformed_dataset = ImagesDataset(path_to_data=data_path)
    train_dl = DataLoader(transformed_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    for batch in train_dl:
        image, label = batch
        print(image)
