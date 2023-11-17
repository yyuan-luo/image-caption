import yaml
from models.encoder import Encoder
import torchvision.transforms as transforms
from data.FlickrDataLoader import get_loader

with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

images_path = config['data']['image_dir']
caption_path = config['data']['caption_file']
batch_size = (int(config['training']['batch_size']))

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    loader, dataset = get_loader(images_path, caption_path, image_transform, batch_size)
    encoder = Encoder()
    encoder.eval()
    for (imgs, captions, seq_len) in loader:
        print(imgs.shape)
        print(captions.shape)
        print(seq_len)
        print(encoder(imgs).shape)
        break
