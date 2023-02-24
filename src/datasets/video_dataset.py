import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.transforms import preprocess, augment
import cv2

class video_dataset(Dataset):
    def __init__(self, video_paths, label_list):
        self.video_paths = video_paths
        self.label_list = label_list
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # Load the video file
        cap = cv2.VideoCapture(self.video_paths[index])
        label = self.label_list[index]

        # Extract the frames from the video
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the frame to RGB format and apply any transforms
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess(frame)
            frames.append(frame)

        # Release the video capture object
        cap.release()

        # # Apply data augmentation
        # frames = augment(frames)

        # Convert the frames to a PyTorch tensor and return
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames, label