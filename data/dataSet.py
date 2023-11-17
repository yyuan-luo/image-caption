import os
from os import walk
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer


def load_captions(path):
    f = open(path, "r")
    f.readline()
    captions = {}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in f:
        image, caption = line.split(",", 1)
        caption = caption.replace('"', '')
        caption_tokens = tokenizer.tokenize(caption)
        caption_tokens = tokenizer.convert_tokens_to_ids(caption_tokens)
        captions[image] = tokenizer.build_inputs_with_special_tokens(caption_tokens)
    f.close()
    return captions


def get_image_names(folder_path):
    for (_, _, filenames) in walk(folder_path):
        return filenames


def load_image(path):
    image = Image.open(path)
    return image


class ImageCaptionDataset(Dataset):
    def __init__(self, image_foler_path, caption_path, transform=None):
        self.image_folder_path = image_foler_path
        self.image_names = get_image_names(image_foler_path)
        self.captions = load_captions(caption_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = load_image(self.image_folder_path + "/" + self.image_names[idx])
        caption = self.captions[os.path.basename(self.image_names[idx])]

        if self.transform:
            image = self.transform(image)

        return image, caption
