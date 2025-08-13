import torch

from nuclei_detection.loader import load_data
from nuclei_detection.training import (
    train_model,
    comprehensive_training_strategy,
)


def train_classification_model(data_path, num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Train a classification model on the tile dataset.

    Args:
        data_path (str): Path to the folder containing the data
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
    """

    print("=" * 80)
    print("Histopathology crop image nuclei detection pipeline")
    print("=" * 80)

    # Step 1: Explore and prepare the data
    print("\n1. Loading data")
    print("-" * 50)

    # Prepare the data with the new function
    data = load_data(data_path)

    train_images = data["train_images"]
    train_labels = data["train_labels"] 
    test_images = data["test_images"]
    test_labels = data["test_labels"]
    val_images = data["val_images"]
    val_labels = data["val_labels"]
    channel_means = data["channel_means"]
    channel_stds = data["channel_stds"]

    print("\n2. Model training")
    print("-" * 50)

    model, history = train_model(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images, 
        test_labels=test_labels,
        val_images=val_images,
        val_labels=val_labels,
        channel_means=channel_means,
        channel_stds=channel_stds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    print("\n3. Model saving")
    print("-" * 50)

    # Save the model
    model_path = "nuclei_classifier.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "history": history
    }, model_path)

    print(f"Model saved to: {model_path}")


def train_with_anti_overfitting(data_path, strategy="regularization", num_epochs=30, 
                               batch_size=16, learning_rate=0.0005, **kwargs):
    """
    Training with anti-overfitting strategies

    Args:
        data_path (str): Path to the labeled data
        strategy (str): 'regularization', 'pseudo_labeling'
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        **kwargs: Additional arguments for the strategy
    """

    print("=" * 80)
    print("TRAINING WITH ANTI-OVERFITTING STRATEGIES")
    print("=" * 80)

    # Execute the selected strategy
    comprehensive_training_strategy(
        data_path=data_path,
        strategy=strategy,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )
