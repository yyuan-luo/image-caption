from data.dataSet import ImageCaptionDataset
import torchvision.transforms as transforms
from utils.padding import DynamicPadding
from torch.utils.data import DataLoader
from models.encoder import Encoder
import yaml

with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

images_path = config['data']['image_dir']
caption_path = config['data']['caption_file']
target_size = config['training']['target_size']
target_size = (int(target_size), int(target_size))

image_transform = transforms.Compose([
    DynamicPadding(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    dataset = ImageCaptionDataset(images_path, caption_path, image_transform)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=lambda x: x)

    encoder = Encoder()
    encoder.eval()
    count = 0
    for batch in data_loader:
        count += 1
        image_features = encoder(batch[0][0])
        print(image_features.shape)
        print(batch[0][1])
        if count == 1:
            break
