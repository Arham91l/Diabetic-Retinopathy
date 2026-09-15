from torchvision import transforms
from transformers import AutoImageProcessor

MODEL_NAME = "microsoft/swinv2-tiny-patch4-window16-256"
IMG_SIZE = 256

_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
MEAN, STD = _processor.image_mean, _processor.image_std


def get_transform():
    """Returns the torchvision transform matching training-time preprocessing."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
