from data.collator import MyCollate
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
