import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nuclei_detection.loader import load_model, load_data
from nuclei_detection.dataset import CustomImageDataset, create_transforms
from nuclei_detection.evaluation import evaluate_model, plot_confusion_matrix

def evaluate_saved_model(data_path, model_path, batch_size=32):
    """Evaluate a saved model on the test dataset"""

    # Load the model
    model, channel_means, channel_stds, device, history = load_model(model_path)
    print(f"Using device: {device}")

    dataset = load_data(data_path)
    test_images = dataset["test_images"]
    test_labels = dataset["test_labels"]
    print(f"Loaded {len(test_images)} test images with labels.")

    # Create the transformations
    transform = create_transforms(channel_means, channel_stds)

    # Create the dataset and dataloader
    test_dataset = CustomImageDataset(test_images, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    print("Evaluating the model...")
    metrics = evaluate_model(model, test_loader, device)

    # Display the results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")

    # Display the confusion matrix
    plot_confusion_matrix(metrics["labels"], metrics["predictions"])

    # Display the training history if available
    if history is not None and not isinstance(history, list):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_losses"], label="Train Loss")
        plt.plot(history["val_losses"], label="Validation Loss")
        plt.title("Loss Evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history["train_accuracies"], label="Train Accuracy")
        plt.plot(history["val_accuracies"], label="Validation Accuracy")
        plt.title("Accuracy evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
