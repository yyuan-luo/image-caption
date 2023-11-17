import torch
import yaml
import torch.nn as nn
import torchvision.transforms as transforms
from models.encoder import Encoder
from models.decoder import Decoder
from data.FlickrDataLoader import get_loader

with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

images_path = config['data']['image_dir']
caption_path = config['data']['caption_file']
batch_size = (int(config['training']['batch_size']))

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    loader, dataset = get_loader(images_path, caption_path, image_transform, batch_size)
    embedding_size = 300
    vocabulary_size = dataset.vocabulary.__len__()
    encoder = Encoder(embedding_size)
    decoder = Decoder(vocabulary_size, embedding_size, vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    encoder.eval()
    decoder.eval()
    for (imgs, captions, seq_len) in loader:
        print("images:", imgs.shape)
        print("captions:", captions.shape)
        img_features = encoder(imgs)
        print("image features:", img_features.shape)
        output = decoder(img_features, captions)
        print("output:", output.shape)
        captions = torch.reshape(captions, (-1,))
        print("caption reshape:", captions)
        print("Loss", criterion(output, captions))
        break
