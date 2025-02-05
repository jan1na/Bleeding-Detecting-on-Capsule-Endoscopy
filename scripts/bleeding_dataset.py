import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class BleedDataset(Dataset):
    def __init__(self, root_dir, mode="RGB", augment_times=10, apply_augmentation=True):
        self.root_dir = root_dir
        self.bleeding_dir = os.path.join(root_dir, "bleeding")
        self.healthy_dir = os.path.join(root_dir, "healthy")
        self.apply_augmentation = apply_augmentation
        self.augment_times = augment_times

        # Separate data lists for controlled augmentation
        self.bleeding_data = [
            (os.path.join(self.bleeding_dir, p), 1)
            for p in os.listdir(self.bleeding_dir)
        ]
        self.healthy_data = [
            (os.path.join(self.healthy_dir, p), 0)
            for p in os.listdir(self.healthy_dir)
        ]

        # Combine both, but only duplicate bleeding images for augmentation
        if apply_augmentation:
            self.data = self.bleeding_data * augment_times + self.healthy_data
        else:
            self.data = self.bleeding_data + self.healthy_data

        self.mode = mode.lower()
        if self.mode not in {"rgb", "gray"}:
            raise ValueError("Invalid mode. Use 'RGB' or 'gray'.")

        # Augmentation settings
        self.augmentation = A.Compose([
            A.RandomScale(scale_limit=0.3, p=0.5),  # Zoom in and out
            A.Rotate(limit=40, p=0.7),  # Rotation
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Blur
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Distortion
            A.Resize(height=224, width=224)  # Resize to a fixed size if needed
        ])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _preprocess_image(image):
        image = image[32:544, 32:544]  # Crop black borders
        image[:48, :48] = 0  # Remove artifacts
        image[:31, 452:] = 0
        return image

    def disable_augmentation(self):
        self.apply_augmentation = False
        self.data = self.bleeding_data + self.healthy_data

    def enable_augmentation(self):
        self.apply_augmentation = True
        self.data = self.bleeding_data * self.augment_times + self.healthy_data

    def get_labels(self):
        if self.apply_augmentation:
            return [1] * len(self.bleeding_data) * self.augment_times + [0] * len(self.healthy_data)
        else:
            return [1] * len(self.bleeding_data) + [0] * len(self.healthy_data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.mode == "gray" else cv2.IMREAD_COLOR)
        image = self._preprocess_image(image)

        # Apply augmentation **only for bleeding images**
        if self.apply_augmentation and label == 1:
            image = self.augmentation(image=image)["image"]
        else:
            image = cv2.resize(image, (224, 224))  # Resize healthy images without augmentation

        # Convert to tensor-friendly format
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # Convert to CxHxW
        else:
            image = image[np.newaxis, ...]
        image = torch.from_numpy(image).float()

        return image, label
