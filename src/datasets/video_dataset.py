import torch
import numpy as np
from torch.utils.data import Dataset
# from datasets.transforms import preprocess, augment
import cv2
from utils.util_fnc import read_video

class video_dataset(Dataset):
    def __init__(self, video_paths, label_list, transform=None):
        self.video_paths = video_paths
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # Load the video file
        frames = read_video(self.video_paths[index])

        frames = self.transform({'video': frames})['video']
        # frames = torch.from_numpy(frames)
        
        if self.label_list is None:
            return frames
        else:
            label = self.label_list[index]
            return frames, label