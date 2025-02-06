import os
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BleedDataset(Dataset):
    def __init__(self, root_dir, mode="RGB", augment_times=8, apply_augmentation=False):
        # Root directory containing the dataset
        self.root_dir = root_dir
        # Directories for bleeding and healthy images
        self.bleeding_dir = os.path.join(root_dir, "bleeding")
        self.healthy_dir = os.path.join(root_dir, "healthy")
        self.apply_augmentation = apply_augmentation  # Flag to apply augmentation
        # Set the number of times to augment bleeding images
        self.augment_times = augment_times if apply_augmentation else 1

        # Create lists of paths for bleeding and healthy images with corresponding labels
        self.bleeding_data = [(os.path.join(self.bleeding_dir, p), 1) for p in os.listdir(self.bleeding_dir)]
        self.healthy_data = [(os.path.join(self.healthy_dir, p), 0) for p in os.listdir(self.healthy_dir)]

        # Combine both lists, duplicating bleeding images for augmentation
        self.data = self.bleeding_data * self.augment_times + self.healthy_data

        # Set image mode (RGB or grayscale)
        self.mode = mode.lower()
        if self.mode not in {"rgb", "gray"}:
            raise ValueError("Invalid mode. Use 'RGB' or 'gray'.")

        # Define the augmentation pipeline
        self.augmentation = A.Compose(
            [A.RandomScale(scale_limit=0.3, p=0.5),  # Zoom in and out
             A.Rotate(limit=40, p=0.7),  # Rotation
             A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Blur
             A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Distortion
             A.Resize(height=224, width=224)  # Resize to a fixed size
             ])

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.data)

    @staticmethod
    def _preprocess_image(image):
        # Crop black borders from the image
        image = image[32:544, 32:544]  # Crop region
        # Remove specific artifacts from the image
        image[:48, :48] = 0  # Clear top-left corner
        image[:31, 452:] = 0  # Clear top-right corner
        return image

    def disable_augmentation(self):
        # Disable augmentation on-the-fly
        self.apply_augmentation = False

    def enable_augmentation(self):
        # Enable augmentation on-the-fly
        self.apply_augmentation = True

    def get_labels(self):
        # Return a list of labels for the entire dataset
        return [1] * len(self.bleeding_data) * self.augment_times + [0] * len(self.healthy_data)

    def __getitem__(self, idx):
        # Retrieve an image and its label from the dataset
        image_path, label = self.data[idx]
        # Read image in the appropriate mode (gray or RGB)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.mode == "gray" else cv2.IMREAD_COLOR)
        # Preprocess the image (crop and remove artifacts)
        image = self._preprocess_image(image)

        # Apply augmentation only to bleeding images
        if self.apply_augmentation and label == 1:
            image = self.augmentation(image=image)["image"]
        else:
            # Resize healthy images without applying augmentation
            image = cv2.resize(image, (224, 224))

        # Convert image to a format suitable for PyTorch (CxHxW)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # Convert from HxWxC to CxHxW
        else:
            # For grayscale images, add the channel dimension
            image = image[np.newaxis, ...]
        # Convert image to a PyTorch tensor
        image = torch.from_numpy(image).float()

        return image, label  # Return the processed image and its label
