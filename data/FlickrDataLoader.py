from data.collator import MyCollate
import torchvision.transforms as transforms
from data.FlickrDataset import FlickrDataset
from torch.utils.data import DataLoader, random_split


def get_loader(image_dir, annotation_file, transform, batch_size=32, num_workers=4, pin_memory=True):
    dataset = FlickrDataset(image_dir, annotation_file, transform)
    pad_idx = dataset.vocabulary.stoi['[PAD]']

    train_size = int(0.6 * dataset.__len__())
    val_size = int(0.2 * dataset.__len__())
    test_size = dataset.__len__() - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return train_loader, val_loader, test_loader, data_loader, dataset


if __name__ == '__main__':
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader, _, test_loader, data_loader, dataset = get_loader('./flickr8k/Images', './flickr8k/captions.txt', image_transform, 4)
    print(len(dataset.vocabulary.itos))
    for imgs, captions, seq_lens in test_loader:
        print(captions.shape, captions)
        print([dataset.vocabulary.itos[caption] for caption in captions[0].detach().cpu().numpy()])
        break
