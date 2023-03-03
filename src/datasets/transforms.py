import torch
import torchvision.transforms as transforms
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomResizedCrop,
    RandomHorizontalFlip
)

def preprocess(frame):
    frame = transforms.ToTensor()(frame)
    
    # Apply your preprocessing steps here
    frame = transforms.Resize((128, 128))(frame)
    frame = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(frame)
    return frame

def augment(frames):
    # Apply your augmentation steps here
    frames = [transforms.RandomHorizontalFlip()(frame) for frame in frames]
    frames = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(frame) for frame in frames]
    return frames

def train_transform():
    _train_transform = Compose(
            [
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                        Lambda(lambda x: torch.from_numpy(x)),
                        UniformTemporalSubsample(num_samples=32),
                        Lambda(lambda x: x / 255.0),
                        ShortSideScale(size=256),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        RandomHorizontalFlip(p=0.5),
                    ]),
            ),
            RemoveKey('orig_video'),  # Remove the original video frames from the transformed data
        ]
    )

    return _train_transform

def eval_transform():
    _eval_transform = Compose(
            [
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                            Lambda(lambda x: torch.from_numpy(x)),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]),
            )])
    return _eval_transform
