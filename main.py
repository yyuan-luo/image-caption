import os
import sys
import math
import yaml
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from models.encoder import Encoder
from models.decoder import Decoder
from utils.plot import plot_loss, plot_test
from data.FlickrDataLoader import get_loader

with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

images_path = config['data']['image_dir']
caption_path = config['data']['caption_file']
training_percentage = float(config['training']['training_percentage'])
batch_size = int(config['training']['batch_size'])
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
    else:
        print("wrong mode assigned")
        sys.exit()
    
    if use_gpu and torch.cuda.is_available():
        print("cuda in use")
    elif use_gpu and (not torch.cuda.is_available()):
        print("cuda not available, cpu in use")
    else:
        print("cpu in use")

    train_loader, test_loader, data_loader, dataset = get_loader(images_path, caption_path, image_transform, batch_size, training_percentage)
    embedding_size = 300
    vocabulary_size = dataset.vocabulary.__len__()
    encoder = Encoder(embedding_size).to(device)
    decoder = Decoder(vocabulary_size, embedding_size, vocabulary_size, hidden_dim=embedding_size, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    total_steps = math.floor(int(dataset.__len__() * training_percentage) / batch_size)

    if is_training:
        # training
        print("training:")
        if (len(args) == 3):
            print(f'Loading encoder-{training_starting_file}.pth and decoder-{training_starting_file}.pth')
            encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'encoder-{training_starting_file}.pth')))
            decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'decoder-{training_starting_file}.pth')))
        encoder.train()
        decoder.train()
        train_loss_epoch = []
        train_loss = []
        params = list(encoder.embedding.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr)
        outer = tqdm(total=num_epochs - training_starting_file, desc="Epoch", position=0, leave=True)
        train_log = tqdm(total=0, position=2, bar_format='{desc}', leave=False)
        for epoch in range(num_epochs - training_starting_file):
            inner = tqdm(total=total_steps, desc="Batch", position=1, leave=False)
            for i_step, (imgs, captions, seq_lens) in enumerate(train_loader):
                encoder.zero_grad()
                decoder.zero_grad()
                imgs = imgs.to(device)
                captions = captions.to(device)
                img_features = encoder(imgs)
                # seq_lens = [seq_len + 1 for seq_len in seq_lens] TODO: How to correctly handle the pre-injected image features
                seq_lens = torch.tensor(seq_lens).to(device)
                output = decoder(img_features, captions, seq_lens)
                captions = torch.reshape(captions, (-1,))
                loss = criterion(output, captions)  # 是将output转化为2994 还是将captions进行embedding呢？
                loss.backward()
                optimizer.step()
                L = loss.item()
                stats = 'Loss: %.4f, Perplexity: %5.4f' % (L, np.exp(L))
                train_log.set_description_str(stats)
                if i_step % log_interval:
                    train_loss.append(L)
                inner.update(1)
            if epoch % log_interval == 0:
                torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, f"encoder-{epoch + 1 + training_starting_file}.pth"))
                torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, f"decoder-{epoch + 1 + training_starting_file}.pth"))
            train_loss_epoch.append(train_loss)
            train_loss = []
            inner.close()
            outer.update(1)
        outer.close()

        # testing
        encoder.eval()
        decoder.eval()
        print("\ntesting:")
        test_loss = []
        total_steps = math.floor(int(dataset.__len__() * (1 - training_percentage)) / batch_size)
        inner = tqdm(total=total_steps, desc="Batch", position=0, leave=True)
        test_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        for i_step, (imgs, captions, seq_lens) in enumerate(test_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            seq_lens = torch.tensor(seq_lens).to(device)
            img_features = encoder(imgs)
            output = decoder(img_features, captions, seq_lens)
            captions = torch.reshape(captions, (-1,))
            loss = criterion(output, captions)
            L = loss.item()
            test_loss.append(L)
            stats = 'Loss: %.4f, Perplexity: %5.4f' % (L, np.exp(L))
            test_log.set_description_str(stats)
            inner.update(1)
        plot_loss(train_loss_epoch[-1], test_loss, log_dir)
    else:
        print(f'Loading encoder-{evaluating_checkpoint_file}.pth and decoder-{evaluating_checkpoint_file}.pth')
        encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'encoder-{evaluating_checkpoint_file}.pth')))
        decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'decoder-{evaluating_checkpoint_file}.pth')))
        encoder.eval()
        decoder.eval()
        image_index_in_batch = 0
        for i_step, (imgs, captions, _) in enumerate(data_loader):
            index = i_step * batch_size
            test_input = imgs[image_index_in_batch].to(device).unsqueeze(0)
            words_indices = decoder.sampler(encoder(test_input))
            words = []
            print(words_indices)
            for word_index in words_indices:
                words.append(dataset.vocabulary.itos[word_index])
            plot_test(words[1:], index + image_index_in_batch, i_step, images_path, caption_path, log_dir)
            if i_step == 10:
                break