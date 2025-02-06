import os
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BleedDataset(Dataset):
    def __init__(self, root_dir, mode="RGB", augment_times=8, apply_augmentation=False):
        # Root directory containing the dataset, which includes "bleeding" and "healthy" subfolders
        self.root_dir = root_dir
        # Paths to the directories containing bleeding and healthy images
        self.bleeding_dir = os.path.join(root_dir, "bleeding")
        self.healthy_dir = os.path.join(root_dir, "healthy")
        self.apply_augmentation = apply_augmentation  # Flag to apply data augmentation
        # Set the number of times to augment bleeding images; used to address class imbalance
        self.augment_times = augment_times if apply_augmentation else 1

        # List the paths of bleeding and healthy images with their corresponding labels (1 for bleeding, 0 for healthy)
        self.bleeding_data = [(os.path.join(self.bleeding_dir, p), 1) for p in os.listdir(self.bleeding_dir)]
        self.healthy_data = [(os.path.join(self.healthy_dir, p), 0) for p in os.listdir(self.healthy_dir)]

        # Combine the data, applying augmentation to the bleeding images by duplicating them
        self.data = self.bleeding_data * self.augment_times + self.healthy_data

        # Define the mode for image loading (RGB or grayscale)
        self.mode = mode.lower()
        if self.mode not in {"rgb", "gray"}:
            raise ValueError("Invalid mode. Use 'RGB' or 'gray'.")

        # Define the augmentation pipeline for bleeding images, such as scaling, rotation, and distortion
        self.augmentation = A.Compose(
            [A.RandomScale(scale_limit=0.3, p=0.5),  # Zoom in and out
             A.Rotate(limit=40, p=0.7),  # Rotation
             A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Blur
             A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Distortion
             A.Resize(height=224, width=224)  # Resize to a fixed size (224x224) for model input
             ])

    def __len__(self):
        # Return the total number of images (including augmented ones) in the dataset
        return len(self.data)

    @staticmethod
    def _preprocess_image(image):
        # Crop the image to remove black borders and unwanted regions
        image = image[32:544, 32:544]  # Crop region to remove unnecessary areas
        # Remove specific artifacts from the image to improve quality
        image[:48, :48] = 0  # Clear top-left corner where artifact might be
        image[:31, 452:] = 0  # Clear top-right corner where artifact might be
        return image

    def disable_augmentation(self):
        # Disable augmentation on-the-fly (useful when testing or evaluating)
        self.apply_augmentation = False

    def enable_augmentation(self):
        # Enable augmentation on-the-fly
        self.apply_augmentation = True

    def get_labels(self):
        # Generate a list of labels for the entire dataset (1 for bleeding, 0 for healthy)
        return [1] * len(self.bleeding_data) * self.augment_times + [0] * len(self.healthy_data)

    def __getitem__(self, idx):
        # Retrieve an image and its corresponding label by index
        image_path, label = self.data[idx]
        # Read image in grayscale if the mode is set to "gray", otherwise in RGB color
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.mode == "gray" else cv2.IMREAD_COLOR)
        # Preprocess the image (crop and remove artifacts)
        image = self._preprocess_image(image)

        # Apply augmentation only to bleeding images (to prevent augmentation on healthy images)
        if self.apply_augmentation and label == 1:
            image = self.augmentation(image=image)["image"]
        else:
            # Resize healthy images without augmentation for consistent input size
            image = cv2.resize(image, (224, 224))

        # Convert image to a PyTorch-compatible format (CxHxW format for RGB, or 1xHxW for grayscale)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # Convert from HxWxC to CxHxW
        else:
            # For grayscale images, add a single channel dimension
            image = image[np.newaxis, ...]
        # Convert the NumPy array to a PyTorch tensor and ensure it's in float format for model input
        image = torch.from_numpy(image).float()

        # Return the processed image and its label for training or evaluation
        return image, label  # Return the processed image and its associated label
