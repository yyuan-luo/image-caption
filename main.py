import sys
import math
import yaml
import torch
import numpy as np
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
lr = float(config['training']['learning_rate'])
num_epochs = int(config['training']['num_epochs'])

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    train_loader, test_loader, data_loader, dataset = get_loader(images_path, caption_path, image_transform, batch_size)
    embedding_size = 300
    vocabulary_size = dataset.vocabulary.__len__()
    encoder = Encoder(embedding_size)
    decoder = Decoder(vocabulary_size, embedding_size, vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    total_steps = math.ceil(int(dataset.__len__() * 0.8) / batch_size)

    # training
    print("training:")
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        for i_step, (imgs, captions, _) in enumerate(train_loader):
            encoder.zero_grad()
            decoder.zero_grad()
            img_features = encoder(imgs)
            mask = (captions > vocabulary_size)
            print()
            print(mask.nonzero())
            output = decoder(img_features, captions)
            captions = torch.reshape(captions, (-1,))
            loss = criterion(output, captions)
            params = list(encoder.embedding.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.Adam(params, lr)
            L = loss.item()
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
                epoch, num_epochs, i_step, total_steps, L, np.exp(L))

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

    # testing
    encoder.eval()
    decoder.eval()
    print("\ntesting:")
    for i_step, (imgs, captions, _) in enumerate(test_loader):
        img_features = encoder(imgs)
        output = decoder(img_features, captions)
        captions = torch.reshape(captions, (-1,))
        loss = criterion(output, captions)
        L = loss.item()
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, num_epochs, i_step, total_steps, L, np.exp(L))

        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()


        # test_input = imgs[0].unsqueeze(0)
        # words_indices = decoder.sampler(encoder(test_input))
        # print(words_indices)
        # words = []
        # for word_index in words_indices:
        #     words.append(dataset.vocabulary.itos[word_index])
        # print(words)
