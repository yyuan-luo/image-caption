from dataSet import ImageCaptionDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def calculate_padding(image, target_size):
    width, height = image.size
    target_width, target_height = target_size

    left_padding = max(0, (target_width - width) // 2)
    right_padding = max(0, target_width - width - left_padding)
    top_padding = max(0, (target_height - height) // 2)
    bottom_padding = max(0, target_height - height - top_padding)

    return left_padding, top_padding, right_padding, bottom_padding


class DynamicPadding:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        padding = calculate_padding(image, self.target_size)
        padding_transforms = transforms.Pad(padding, fill=0)
        return padding_transforms(image)


folder_path = "./data/flickr8k/images"
caption_path = "./data/flickr8k/captions.txt"
target_size = (500, 500)

image_transform = transforms.Compose([
    DynamicPadding(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    dataset = ImageCaptionDataset(folder_path, caption_path, image_transform)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=lambda x: x)

    for batch in data_loader:
        print(batch[0])