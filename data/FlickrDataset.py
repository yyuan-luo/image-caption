import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from data.vocabulary import Vocabulary

'''
Work highly relies on https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-8-working-with-caption-dataset
'''


class FlickrDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, frequency_threshold=5):
        self.image_dir = image_dir
        self.transform = transform
        self.frequency_threshold = frequency_threshold
        self.df = pd.read_csv(annotation_file)
        self.images = self.df["image"]
        self.captions = self.df["caption"]
        self.vocabulary = Vocabulary(self.frequency_threshold)
        self.vocabulary.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_name = self.images[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")     # PIL image B->G->R

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocabulary.stoi['[SOS]']]
        numericalized_caption += self.vocabulary.numericalize(caption)
        numericalized_caption.append(self.vocabulary.stoi['[EOS]'])

        return image, torch.tensor(numericalized_caption)
