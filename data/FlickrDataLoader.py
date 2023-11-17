from data.collator import MyCollate
from torch.utils.data import DataLoader
from data.FlickrDataset import FlickrDataset


def get_loader(image_dir, annotation_file, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    dataset = FlickrDataset(image_dir, annotation_file, transform)

    pad_idx = dataset.vocabulary.stoi['[PAD]']

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, dataset
