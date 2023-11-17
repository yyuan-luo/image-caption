import os
import sys
import math
import yaml
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from utils.plot import plot_loss, plot_test
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
    args = sys.argv
    is_training = True
    training_starting_file = 0
    evaluating_checkpoint_file = 1
    if args[1] == 'training':
        print("training mode, starting from 0 unless specified")
        if (len(args) == 3):
            training_starting_file = int(args[2])
    elif args[1] == 'evaluating':
        is_training = False
        print("evaluating mode, default checkpoint files: encoder/decoder-1.pth unless specified")
        if len(args) == 3:
            evaluating_checkpoint_file = int(args[2])
     
    if use_gpu and torch.cuda.is_available():
        print("cuda in use")
    elif use_gpu and (not torch.cuda.is_available()):
        print("cuda not available, cpu in use")
    else:
        print("cpu in use")

    train_loader, test_loader, data_loader, dataset = get_loader(images_path, caption_path, image_transform, batch_size)
    embedding_size = 300
    vocabulary_size = dataset.vocabulary.__len__()
    encoder = Encoder(embedding_size).to(device)
    decoder = Decoder(vocabulary_size, embedding_size, vocabulary_size, hidden_dim=embedding_size, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    total_steps = math.floor(int(dataset.__len__() * 0.8) / batch_size)

    if is_training:
        # training
        print("training:")
        if (len(args) == 3):
            encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'encoder-{training_starting_file}.pth')))
            decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'decoder-{training_starting_file}.pth')))
        encoder.train()
        decoder.train()
        train_loss_epoch = []
        train_loss = []
        for epoch in range(num_epochs - training_starting_file):
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
                        (epoch + 1 + training_starting_file), num_epochs, i_step, total_steps, L, np.exp(L))
                    print('\r' + stats, end="")
                    sys.stdout.flush()
            torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, f"encoder-{epoch + 1 + training_starting_file}.pth"))
            torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, f"decoder-{epoch + 1 + training_starting_file}.pth"))
            train_loss_epoch.append(train_loss)
            train_loss = []

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
        plot_loss(train_loss_epoch[-1], test_loss, log_dir)
    else:
        print(f'Loading encoder-{evaluating_checkpoint_file}.pth and decoder-{evaluating_checkpoint_file}.pth')
        encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'encoder-{evaluating_checkpoint_file}.pth')))
        decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'decoder-{evaluating_checkpoint_file}.pth')))
        encoder.eval()
        decoder.eval()
        for i_step, (imgs, captions, _) in enumerate(data_loader):
            index = i_step * batch_size
            test_input = imgs[0].to(device).unsqueeze(0)
            words_indices = decoder.sampler(encoder(test_input))
            words = []
            print(words_indices)
            for word_index in words_indices:
                words.append(dataset.vocabulary.itos[word_index])
            plot_test(words[1:], index, i_step, images_path, caption_path, log_dir)
            if i_step == 10:
                break