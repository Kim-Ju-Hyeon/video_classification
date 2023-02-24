import torchvision.transforms as transforms

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
