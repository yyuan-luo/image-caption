import os
import sys
import math
import yaml
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from utils.plot import plot_loss
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
use_gpu = config['other']['use_gpu']
log_interval = int(config['other']['log_interval'])
log_dir = config['results']['log_dir']
checkpoint_dir = config['results']['checkpoint_dir']

device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    if use_gpu and torch.cuda.is_available():
        print("cuda in use")
    elif use_gpu and (not torch.cuda.is_available()):
        print("Cuda not available, cpu in use")
    else:
        print("cpu in use")

    train_loader, test_loader, data_loader, dataset = get_loader(images_path, caption_path, image_transform, batch_size)
    embedding_size = 300
    vocabulary_size = dataset.vocabulary.__len__()
    encoder = Encoder(embedding_size).to(device)
    decoder = Decoder(vocabulary_size, embedding_size, vocabulary_size, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    total_steps = math.floor(int(dataset.__len__() * 0.8) / batch_size)

    # training
    print("training:")
    encoder.train()
    decoder.train()
    train_loss = []
    for epoch in range(num_epochs):
        for i_step, (imgs, captions, _) in enumerate(train_loader):
            encoder.zero_grad()
            decoder.zero_grad()
            imgs = imgs.to(device)
            captions = captions.to(device)
            img_features = encoder(imgs)
            output = decoder(img_features, captions)
            captions = torch.reshape(captions, (-1,))
            loss = criterion(output, captions)
            params = list(encoder.embedding.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.Adam(params, lr)
            loss.backward()
            optimizer.step()
            L = loss.item()
            if i_step % log_interval:
                train_loss.append(L)
                stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
                    epoch, num_epochs, i_step, total_steps, L, np.exp(L))
                print('\r' + stats, end="")
                sys.stdout.flush()
        torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, "encoder-%d.pth" % epoch))
        torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, "decoder-%d.pth" % epoch))

    # testing
    encoder.eval()
    decoder.eval()
    print("\ntesting:")
    test_loss = []
    total_steps = math.floor(int(dataset.__len__() * 0.2) / batch_size)
    for i_step, (imgs, captions, _) in enumerate(test_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        img_features = encoder(imgs)
        output = decoder(img_features, captions)
        captions = torch.reshape(captions, (-1,))
        loss = criterion(output, captions)
        L = loss.item()
        test_loss.append(L)
        stats = 'Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (i_step, total_steps, L, np.exp(L))
        print('\r' + stats, end="")
        sys.stdout.flush()
    print()
    plot_loss(train_loss, test_loss, log_dir)


        # test_input = imgs[0].unsqueeze(0)
        # words_indices = decoder.sampler(encoder(test_input))
        # print(words_indices)
        # words = []
        # for word_index in words_indices:
        #     words.append(dataset.vocabulary.itos[word_index])
        # print(words)
