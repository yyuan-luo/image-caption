import torchvision.transforms as transforms


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
