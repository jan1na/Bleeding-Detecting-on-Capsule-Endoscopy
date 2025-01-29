import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class BleedDataset(Dataset):
    def __init__(self, root_dir, mode="RGB", augment_times=10):
        self.root_dir = root_dir
        self.bleeding_dir = os.path.join(root_dir, "bleeding")
        self.healthy_dir = os.path.join(root_dir, "healthy")

        # Combine images and labels into tuples
        self.data = [
            (os.path.join(self.bleeding_dir, p), 1)
            for p in os.listdir(self.bleeding_dir)
        ] + [
            (os.path.join(self.healthy_dir, p), 0)
            for p in os.listdir(self.healthy_dir)
        ]

        self.mode = mode.lower()
        if self.mode not in {"rgb", "gray"}:
            raise ValueError("Invalid mode. Use 'RGB' or 'gray'.")

        # Augmentation settings
        self.augment_times = augment_times
        self.augmentation = A.Compose([
            A.RandomScale(scale_limit=0.2, p=0.5),  # Zoom in and out
            A.Rotate(limit=40, p=0.7),  # Rotation
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Color and intensity change
            # A.GaussianNoise(var_limit=(10.0, 50.0), p=0.5)  # Additive noise
            A.Resize(height=512, width=512)  # Resize to a fixed size if needed
        ])

    def __len__(self):
        return len(self.data) * self.augment_times  # Increased size for augmented data

    @staticmethod
    def _preprocess_image(image):
        image = image[32:544, 32:544]  # Crop black borders
        image[:48, :48] = 0  # Remove artifacts
        image[:31, 452:] = 0
        return image

    def __getitem__(self, idx):
        image_path, label = self.data[idx % len(self.data)]  # To ensure repeating original data
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.mode == "gray" else cv2.IMREAD_COLOR)
        image = self._preprocess_image(image)

        # Apply augmentation
        augmented_image = self.augmentation(image=image)["image"]

        # Convert to tensor-friendly format
        if augmented_image.ndim == 3:
            augmented_image = np.transpose(augmented_image, (2, 0, 1))  # Convert to CxHxW
        else:
            augmented_image = augmented_image[np.newaxis, ...]
        augmented_image = torch.from_numpy(augmented_image).float()

        return augmented_image, label
