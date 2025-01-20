import os
import sys
import math
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from models.res2rnn import Res2RNN
from utils.plot import plot_loss, plot_test
from data.FlickrDataLoader import get_loader

with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

images_path = config['data']['image_dir']
caption_path = config['data']['caption_file']
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
    parser = argparse.ArgumentParser(description="Image Caption with RNN")
    parser.add_argument('-m', '--mode', help="choose to train or evaluate the model (training/evaluating)",
                        required=True)
    parser.add_argument('-l', '--load', help="select which checkpoint to load for continuing training or evaluating",
                        required=False)
    args = vars(parser.parse_args())
    is_training = True
    training_starting_file = 0
    evaluating_checkpoint_file = 1
    if args['mode'] == 'training':
        print("training mode: training starts from ground unless checkpoint file specified")
    elif args['mode'] == 'evaluating':
        is_training = False
    else:
        print("wrong mode assigned")
        sys.exit()

    if use_gpu and torch.cuda.is_available():
        print("cuda in use")
    elif use_gpu and (not torch.cuda.is_available()):
        print("cuda not available, cpu in use")
    else:
        print("cpu in use")

    train_loader, val_loader, test_loader, data_loader, dataset = get_loader(images_path, caption_path, image_transform,
                                                                             batch_size)
    embedding_size = 256
    vocabulary_size = dataset.vocabulary.__len__()
    model = Res2RNN(vocabulary_size, embedding_size)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocabulary.stoi['[PAD]'])  # ignore <PAD>
    total_steps = math.floor(int(dataset.__len__() * 0.6) / batch_size)

    if is_training:
        print("training:")
        model.train()
        train_loss_epoch = []
        train_loss = []
        params = list(model.embedding.parameters())
        optimizer = torch.optim.Adam(params, lr)
        outer = tqdm(total=num_epochs - training_starting_file, desc="Epoch", position=0, leave=True)
        train_log = tqdm(total=0, position=2, bar_format='{desc}', leave=False)
        val_best = 100
        for epoch in range(num_epochs - training_starting_file):
            inner = tqdm(total=total_steps, desc="Batch", position=1, leave=False)
            for i_step, (imgs, captions, seq_lens) in enumerate(train_loader):
                model.zero_grad()
                imgs = imgs.to(device)
                captions = captions.to(device)
                seq_lens = torch.tensor(seq_lens).to(device)
                output = model(imgs, captions[:, :-1], seq_lens)
                captions = captions.reshape(-1)
                loss = criterion(torch.reshape(output, (-1, )), captions)  # Compute loss

                loss.backward()
                optimizer.step()
                L = loss.item()
                stats = 'Loss: %.4f, Perplexity: %5.4f' % (L, np.exp(L))
                train_log.set_description_str(stats)
                if i_step % log_interval:
                    train_loss.append(L)
                inner.update(1)
            # validating
            val_loss = []
            for i_step, (imgs, captions, seq_lens) in enumerate(val_loader):
                imgs = imgs.to(device)
                captions = captions.to(device)
                seq_lens = torch.tensor(seq_lens).to(device)
                output = model(imgs, captions, seq_lens)
                captions = torch.reshape(captions, (-1,))
                loss = criterion(output, captions)
                L = loss.item()
                val_loss.append(L)
            average_loss = sum(val_loss) / len(val_loss)
            if average_loss < val_best:
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, f"model-best.pth"))
                val_best = average_loss
            train_loss_epoch.append(train_loss)
            train_loss = []
            inner.close()
            outer.update(1)
        outer.close()


        # testing
        model.eval()
        print("\ntesting:")
        test_loss = []
        total_steps = math.floor(int(dataset.__len__() * 0.2) / batch_size)
        inner = tqdm(total=total_steps, desc="Batch", position=0, leave=True)
        test_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        for i_step, (imgs, captions, seq_lens) in enumerate(test_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            seq_lens = torch.tensor(seq_lens).to(device)
            output = model(imgs, captions, seq_lens)
            captions = torch.reshape(captions, (-1,))
            loss = criterion(output, captions)
            L = loss.item()
            test_loss.append(L)
            stats = 'Loss: %.4f, Perplexity: %5.4f' % (L, np.exp(L))
            test_log.set_description_str(stats)
            inner.update(1)
        plot_loss(train_loss_epoch[-1], test_loss, log_dir)
    else:
        print(f'Loading model-best.pth')
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'model-best.pth')))
        model.eval()
        image_index_in_batch = 0
        for i_step, (imgs, captions, _) in enumerate(data_loader):
            index = i_step * batch_size
            test_input = imgs[image_index_in_batch].to(device).unsqueeze(0)
            words_indices = model.decoder.sampler(model.encoder(test_input))
            words = []
            print(words_indices)
            for word_index in words_indices:
                words.append(dataset.vocabulary.itos[word_index])
            plot_test(words[1:], index + image_index_in_batch, i_step, images_path, caption_path, log_dir)
            if i_step == 10:
                break
