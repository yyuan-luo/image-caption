from dataSet import ImageCaptionDataset

folder_path = "./data/flickr8k/images"
caption_path = "./data/flickr8k/captions.txt"

dataset = ImageCaptionDataset(folder_path, caption_path)

print(dataset.max_width, dataset.max_height)