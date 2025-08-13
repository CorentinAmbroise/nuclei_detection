import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


def explore_dataset(data_path):
    """
    Explore the dataset by displaying basic statistics and visualizations.
    
    Parameters:
        data_path (str): Path to the folder containing the dataset.
    """
    labels = pd.read_csv(os.path.join(data_path, "labels.csv"))
    print("Labels overview:")
    print(labels.head())

    print("\nLabels distribution:")
    print(labels.label.value_counts())

    # Load the dataset
    image_files = [name for name in os.listdir(data_path) if name.endswith(".png")]

    # Load images into a numpy array
    images = np.zeros((len(image_files), 256, 256, 3), dtype=np.uint32)
    for file in image_files:
        img_path = os.path.join(data_path, file)
        tile_id = int(file.split(".")[0].split("_")[-1])
        img = np.array(Image.open(img_path).convert("RGB"))  # Ensure image is RGB
        images[tile_id] = img

    print(len(images), "images loaded.")

    tile_ids = np.arange(len(images))
    labeled_images = images[labels.tile_id.values]
    unlabeled_idx = np.setdiff1d(tile_ids, labels.tile_id.values)
    filtered_labels = labels.label.values

    print(f"Found {len(labeled_images)} labeled images out of {len(labels)} labels.")

    print(images.min(), "minimum pixel value.")
    print(images.max(), "maximum pixel value.")

    # Create subplots for better histogram visualization
    _, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Plot each channel separately
    colors = ["red", "green", "blue"]
    for idx in range(3):
        channel_data = (images[:, :, :, idx]).ravel()
        axes[idx].hist(channel_data, bins=30, color=colors[idx], alpha=0.7)
        axes[idx].set_title(f"{colors[idx].capitalize()} Channel")
        axes[idx].set_xlabel("Pixel Value")
        axes[idx].set_ylabel("Frequency")

    # Plot all channels overlapped
    for idx in range(3):
        channel_data = (images[:, :, :, idx]).ravel()
        axes[3].hist(channel_data, bins=30, color=colors[idx], alpha=0.5,
                     label=f"{colors[idx].capitalize()}")
    axes[3].set_title("All Channels Overlapped")
    axes[3].set_xlabel("Pixel Value")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    print("Computing statistics on the dataset...")
    print("images shape:", images.shape)
    print("images dtype:", images.dtype)

    channel_means = images.mean(axis=(0, 1, 2))  # Mean across all images and pixels
    channel_stds = images.std(axis=(0, 1, 2))
    print("Full dataset channel-wise means:", channel_means)
    print("Full dataset channel-wise standard deviations:", channel_stds)

    normalized_images = (images - channel_means) / channel_stds
    print("Normalized images shape:", normalized_images.shape)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_images.reshape(len(images), -1))

    labels_to_colors = {
        "NO_NUCLEUS": "blue",
        "NUCLEI": "red",
    }
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[unlabeled_idx, 0], pca_result[unlabeled_idx, 1],
                alpha=0.3, c="gray", label="Unlabeled")

    # Plot labeled data points by category for proper legend
    for label, color in labels_to_colors.items():
        mask = filtered_labels == label
        if np.any(mask):
            plt.scatter(pca_result[labels.tile_id.values[mask], 0], 
                       pca_result[labels.tile_id.values[mask], 1], 
                       alpha=0.5, c=color, label=label)

    plt.title("PCA Result")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    plt.show()
