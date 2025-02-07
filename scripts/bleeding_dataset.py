import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BleedDataset(Dataset):
    """
    Custom PyTorch dataset for loading and preprocessing the bleeding dataset.
    Handles both healthy and bleeding images, with optional data augmentation.
    """

    def __init__(self, root_dir: str, mode: str = "RGB", augment_times: int = 8,
                 apply_augmentation: bool = False) -> None:
        """
        Initialize the dataset with paths, labels, and augmentation settings.

        :param root_dir: Root directory containing the dataset with "bleeding" and "healthy" subfolders.
        :param mode: Image loading mode ("RGB" or "gray").
        :param augment_times: Number of times to duplicate bleeding images for augmentation.
        :param apply_augmentation: Whether to apply data augmentation.
        """
        self.root_dir = root_dir
        self.bleeding_dir = os.path.join(root_dir, "bleeding")
        self.healthy_dir = os.path.join(root_dir, "healthy")

        self.apply_augmentation = apply_augmentation
        self.augment_times = augment_times if apply_augmentation else 1

        self.bleeding_data = [(os.path.join(self.bleeding_dir, p), 1) for p in os.listdir(self.bleeding_dir)]
        self.healthy_data = [(os.path.join(self.healthy_dir, p), 0) for p in os.listdir(self.healthy_dir)]
        self.data = self.bleeding_data * self.augment_times + self.healthy_data

        self.mode = mode.lower()
        if self.mode not in {"rgb", "gray"}:
            raise ValueError("Invalid mode. Use 'RGB' or 'gray'.")

        self.augmentation = A.Compose([
            A.RandomScale(scale_limit=0.3, p=0.5),
            A.Rotate(limit=40, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.Resize(height=224, width=224)
        ])

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        :return: Number of images including augmented samples.
        """
        return len(self.data)

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image by cropping and removing artifacts.

        :param image: Input image as a NumPy array.
        :return: Processed image with artifacts removed.
        """
        image = image[32:544, 32:544]  # Crop black borders
        image[:48, :48] = 0  # Remove artifacts
        image[:31, 452:] = 0
        return image

    def disable_augmentation(self) -> None:
        """
        Disable data augmentation dynamically.
        """
        self.apply_augmentation = False

    def enable_augmentation(self) -> None:
        """
        Enable data augmentation dynamically.
        """
        self.apply_augmentation = True

    def get_labels(self) -> list[int]:
        """
        Retrieve the labels for all dataset images.

        :return: List of labels (1 for bleeding, 0 for healthy).
        """
        return [1] * len(self.bleeding_data) * self.augment_times + [0] * len(self.healthy_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retrieve a preprocessed image and its label by index.

        :param idx: Index of the image in the dataset.
        :return: Tuple containing the image tensor and its label.
        """
        image_path, label = self.data[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.mode == "gray" else cv2.IMREAD_COLOR)
        image = self._preprocess_image(image)

        if self.apply_augmentation and label == 1:
            image = self.augmentation(image=image)["image"]
        else:
            image = cv2.resize(image, (224, 224))

        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # Convert to CxHxW format
        else:
            image = image[np.newaxis, ...]

        return torch.from_numpy(image).float(), label
