import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from nuclei_detection.utils import get_device
from nuclei_detection.model import LeNet5


def load_data(data_path, validation=False):
    """
    Explore the dataset and optionally return prepared data for training.

    Parameters:
    data_path (str): Path to the dataset directory
    validation (bool): Whether to split the training data into training and validation sets

    Returns:
    dict: returns dict with train/test data and statistics
    """
    labels = pd.read_csv(os.path.join(data_path, "labels.csv"))

    # Load the dataset
    image_files = [name for name in os.listdir(data_path) if name.endswith(".png")]

    # Create an array to hold image data
    images = np.zeros((len(image_files), 256, 256, 3), dtype=np.uint32)
    for file in image_files:
        img_path = os.path.join(data_path, file)
        tile_id = int(file.split(".")[0].split("_")[-1])
        img = np.array(Image.open(img_path).convert("RGB"))  # Ensure image is RGB
        images[tile_id] = img

    print(len(images), "images loaded.")

    tile_ids = np.arange(len(images))
    labeled_images = images[labels.tile_id.values]
    unlabeled_images = images[np.setdiff1d(tile_ids, labels.tile_id.values)]
    label_values = labels.label.values

    # Split the dataset into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        labeled_images, label_values, test_size=0.2,
        random_state=42, stratify=label_values
    )
    val_images, val_labels = None, None
    if validation:
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2,
            random_state=42, stratify=train_labels
        )
    print(len(train_images), "training images.")
    print(len(test_images), "testing images.")
    if validation:
        print(len(val_images), "validation images.")
    print(np.unique(train_labels, return_counts=True))
    print(np.unique(test_labels, return_counts=True))
    print(np.unique(val_labels, return_counts=True))

    print(train_images.shape, "shape of training images.")

    # Compute channelwise means and standard deviations on training set
    channel_means = np.concatenate([
        train_images,
        unlabeled_images], axis=0
    ).mean(axis=(0, 1, 2))
    channel_stds = np.concatenate([
        train_images,
        unlabeled_images], axis=0
    ).std(axis=(0, 1, 2))
    print("Training set channel-wise means:", channel_means)
    print("Training set channel-wise standard deviations:", channel_stds)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "unlabeled_images": unlabeled_images,
        "test_images": test_images,
        "test_labels": test_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "labeled_images": labeled_images,
        "label_values": label_values
    }


def load_model(model_path):
    """
    Load a pre-trained model from the specified path.

    Parameters:
    model_path (str): Path to the saved model file

    Returns:
    model: Loaded model
    """

    device = get_device()
    # Load the model state dict
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # Assuming the model architecture is defined elsewhere
    model = LeNet5(num_classes=2).to(device)

    # Load the state dict into the model
    model.load_state_dict(checkpoint["model_state_dict"])

    channel_means = checkpoint["channel_means"]
    channel_stds = checkpoint["channel_stds"]
    history = checkpoint.get("history", None)

    return model, channel_means, channel_stds, device, history
