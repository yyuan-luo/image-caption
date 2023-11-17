import torch
import os
from os import walk
from PIL import Image
from torch.utils.data import Dataset

def load_captions(path):
    f = open(path, "r")
    f.readline()
    captions = {}
    for line in f:
        key, value = line.split(",", 1)
        captions[key] = value
    f.close()
    return captions


def get_image_names(folder_path):
    for (_, _, filenames) in walk(folder_path):
        return filenames


def load_image(path):
    image = Image.open(path)
    return image

def get_max_width_height(folder_path, image_names):
    max_width, max_height = 0, 0
    for file in image_names:
        image = Image.open(folder_path + "/" + file)
        width, height = image.size
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
    return max_width, max_height


class ImageCaptionDataset(Dataset):
    def __init__(self, image_foler_path, caption_path, transform=None):
        self.image_folder_path = image_foler_path
        self.image_names = get_image_names(image_foler_path)
        self.captions = load_captions(caption_path)
        self.max_width, self.max_height = get_max_width_height(self.image_folder_path, self.image_names)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = load_image(self.image_folder_path + "/" + self.image_names[idx])
        caption = self.captions[os.path.basename(self.image_names[idx])]

        if self.transform:
            image = self.transform(image)

        return image, caption
