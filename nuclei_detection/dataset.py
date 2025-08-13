import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    """Custom dataset for loading images and labels"""

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        # Convert labels to numeric format if provided
        self.labels_numeric = None
        if labels is not None:
            self.label_to_idx = {"NO_NUCLEUS": 0, "NUCLEI": 1}
            self.labels_numeric = np.array([self.label_to_idx[label] for label in labels])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels_numeric[idx] if self.labels_numeric is not None else -1

        if self.transform:
            image = self.transform(image.astype(np.uint8)) # Cast to uint8 for transforms.ToTensor

        return image, torch.tensor(label, dtype=torch.long)


def create_transforms(channel_means, channel_stds):
    """Create normalization transforms for the dataset"""

    # Convert means and stds to 0-1 range for normalization
    channel_means = (np.array(channel_means) / 255).tolist()
    channel_stds = (np.array(channel_stds) / 255).tolist()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_means, std=channel_stds),
    ])

    return transform


def create_augmented_transforms(channel_means, channel_stds):
    """Create transforms with data augmentation for training"""

    # Convert means and stds to 0-1 range for normalization
    channel_means = (np.array(channel_means) / 255).tolist()
    channel_stds = (np.array(channel_stds) / 255).tolist()

    transform = transforms.Compose([
        transforms.ToTensor(),

        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),

        # Color augmentations
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),

        # Spatial augmentations
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),

        # Normalisation
        transforms.Normalize(mean=channel_means, std=channel_stds)
    ])

    return transform


def create_transforms_with_cutout(channel_means, channel_stds):
    """Create transforms with Cutout for training"""

    # Convert means and stds to 0-1 range for normalization
    channel_means = (np.array(channel_means) / 255).tolist()
    channel_stds = (np.array(channel_stds) / 255).tolist()

    transform = transforms.Compose([
        transforms.ToTensor(),

        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),

        # Normalisation
        transforms.Normalize(mean=channel_means, std=channel_stds),

        # Cutout/Random Erasing
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value="random"
        )
    ])

    return transform
