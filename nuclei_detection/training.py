import time
import numpy as np
import copy

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


from nuclei_detection.dataset import (
    CustomImageDataset,
    create_transforms,
    create_augmented_transforms,
    create_transforms_with_cutout
)
from nuclei_detection.model import LeNet5
from nuclei_detection.evaluation import evaluate_model, plot_confusion_matrix
from nuclei_detection.loader import load_data, load_model
from nuclei_detection.utils import get_device


def train_model(train_images, train_labels, test_images, test_labels,
                channel_means, channel_stds, val_images=None,
                val_labels=None, num_epochs=20, batch_size=32, 
                learning_rate=0.001, device=None):
    """Main training loop"""

    # Device configuration
    if device is None:
        device = get_device()
    print(f"Device usage: {device}")

    # Initialize transformations
    transform = create_transforms(channel_means, channel_stds)

    # Create datasets
    train_dataset = CustomImageDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomImageDataset(test_images, test_labels, transform=transform)
    if val_images is not None:
        val_dataset = CustomImageDataset(val_images, val_labels, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = test_loader
    # Initialize the model
    model = LeNet5(num_classes=2).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(f"Training start...")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    if val_images is not None:
        print(f"Validation dataset size: {len(val_dataset)}")
    print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Progress display
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Evaluate on the validation dataset
        val_metrics = evaluate_model(model, val_loader, device)

        # Update learning rate
        # scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])

        # Display epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Validation Acc: {val_metrics['accuracy']:.4f}")
        print(f"Validation Precision: {val_metrics['precision']:.4f}, "
              f"Validation Recall: {val_metrics['recall']:.4f}, "
              f"Validation F1: {val_metrics['f1']:.4f}")
        # print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)

    print("Training completed!")

    # Final evaluation
    final_test_metrics = evaluate_model(model, test_loader, device)
    print("\nFinal metrics:")
    print(f"Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"Precision: {final_test_metrics['precision']:.4f}")
    print(f"Recall: {final_test_metrics['recall']:.4f}")
    print(f"F1-Score: {final_test_metrics['f1']:.4f}")

    # Display confusion matrix
    plot_confusion_matrix(final_test_metrics["labels"], final_test_metrics["predictions"])

    # Metrics plots
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy during training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model, {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "final_metrics": final_test_metrics
    }


def train_model_with_regularization(train_images, train_labels, test_images, test_labels, 
                                   channel_means, channel_stds, val_images=None,
                                   val_labels=None, num_epochs=50, batch_size=16, 
                                   learning_rate=0.001, weight_decay=1e-4, label_smoothing=0, 
                                   device=None, dropout_rate=0.5, use_data_augmentation=True,
                                   use_cutout=False, use_early_stopping=True, patience=10,
                                   min_delta=0.001, verbose=1):
    """
    Training loop with anti-overfitting strategies

    Args:
        use_data_augmentation (bool): Use data augmentation
        use_early_stopping (bool): Use early stopping
        patience (int): Number of epochs without improvement before stopping
        min_delta (float): Minimum improvement considered significant
        weight_decay (float): L2 regularization
        dropout_rate (float): Dropout rate for the model
        verbose (int): Verbosity level (0: silent, 1: progress bar, 2: detailed logging and plotting)
    """

    # Device configuration
    if device is None:
        device = get_device()
    if verbose > 0:
        print(f"Device usage: {device}")
        print(f"Data augmentation: {use_data_augmentation}")
        print(f"Cutout augmentation: {use_cutout}")
        print(f"Early stopping: {use_early_stopping} (patience={patience})")
        print(f"Weight decay: {weight_decay}")

    # Create transformations
    train_transform = create_transforms(channel_means, channel_stds)
    val_transform = create_transforms(channel_means, channel_stds)
    if use_data_augmentation:
        train_transform = create_augmented_transforms(channel_means, channel_stds)
    if use_cutout:
        train_transform = create_transforms_with_cutout(channel_means, channel_stds)


    # Create datasets
    train_dataset = CustomImageDataset(train_images, train_labels, transform=train_transform)
    test_dataset = CustomImageDataset(test_images, test_labels, transform=val_transform)
    if val_images is not None:
        val_dataset = CustomImageDataset(val_images, val_labels, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2)
    if val_images is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model with more regularization
    model = LeNet5(num_classes=2, dropout_rate=dropout_rate).to(device)

    # Loss function and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # More sophisticated scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # Variables for early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []

    if verbose > 1:
        print(f"Training start...")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        if val_images is not None:
            print(f"Validation dataset size: {len(val_dataset)}")
        print("-" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Less frequent progress display
            if (batch_idx + 1) % 20 == 0 and verbose > 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Evaluate on the validation dataset
        val_metrics = evaluate_model(model, val_loader, device)

        # Update learning rate based on validation loss
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])
        learning_rates.append(current_lr)

        # Early stopping check
        if use_early_stopping:
            if val_metrics["loss"] < best_val_loss - min_delta:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                # Save the best model with deep copy
                best_model_state = copy.deepcopy(model.state_dict())
                if verbose > 1:
                    print(f"✓ New best model (val_loss: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if verbose > 1:
                    print(f"⚠ No improvement for {epochs_without_improvement} epochs")

        # Print epoch results
        epoch_time = time.time() - start_time
        if verbose > 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Temps: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}, Validation Acc: {val_metrics['accuracy']:.4f}")
            print(f"Validation Precision: {val_metrics['precision']:.4f}, Validation Recall: {val_metrics['recall']:.4f}, Validation F1: {val_metrics['f1']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print("-" * 60)

        # Check early stopping
        if use_early_stopping and epochs_without_improvement >= patience:
            if verbose > 1:
                print(f"🛑 Early stopping after {epoch+1} epochs (patience={patience})")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                if verbose > 1:
                    print("✓ Model restored to its best state")
            break
        elif epoch == num_epochs - 1:
            # If we reach the last epoch, also restore the best model if available
            if use_early_stopping and best_model_state is not None:
                if verbose > 1:
                    print(f"🏁 Training completed - restoring best model (val_loss: {best_val_loss:.4f})")
                model.load_state_dict(best_model_state)
                if verbose > 1:
                    print("✓ Model restored to its best state")

    if verbose > 1:
        print("Training over!")

    # Final evaluation on the best model
    final_test_metrics = evaluate_model(model, test_loader, device)

    # If verbose > 1, print final metrics and plots
    if verbose > 1:
        print("\nFinal metrics (best model):")
        print(f"Accuracy: {final_test_metrics['accuracy']:.4f}")
        print(f"Precision: {final_test_metrics['precision']:.4f}")
        print(f"Recall: {final_test_metrics['recall']:.4f}")
        print(f"F1-Score: {final_test_metrics['f1']:.4f}")

        # Display confusion matrix
        plot_confusion_matrix(final_test_metrics["labels"], final_test_metrics["predictions"])

        # Metric plots with learning rate
        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0,0].plot(train_losses, label="Train Loss")
        axes[0,0].plot(val_losses, label="Validation Loss")
        axes[0,0].set_title("Évolution de la Loss")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")
        axes[0,0].legend()
        axes[0,0].grid(True)

        # Accuracy
        axes[0,1].plot(train_accuracies, label="Train Accuracy")
        axes[0,1].plot(val_accuracies, label="Validation Accuracy")
        axes[0,1].set_title("Accuracy during training")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Accuracy")
        axes[0,1].legend()
        axes[0,1].grid(True)

        # Learning Rate
        axes[1,0].plot(learning_rates, label="Learning Rate")
        axes[1,0].set_title("Learning Rate Evolution")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Learning Rate")
        axes[1,0].set_yscale("log")
        axes[1,0].legend()
        axes[1,0].grid(True)

        # Overfitting indicator (train/val gap)
        overfitting_gap = np.array(train_accuracies) - np.array(val_accuracies)
        axes[1,1].plot(overfitting_gap, label="Train-Val Gap", color="red")
        axes[1,1].set_title("Overfitting Indicator (Train-Val Gap)")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Accuracy Gap")
        axes[1,1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1,1].legend()
        axes[1,1].grid(True)

        plt.tight_layout()
        plt.show()

    return model, {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
        "final_metrics": final_test_metrics,
        "best_val_loss": best_val_loss,
        "stopped_early": epochs_without_improvement >= patience if use_early_stopping else False
    }


def pseudo_labeling_strategy(model, unlabeled_images, channel_means, channel_stds, 
                            confidence_threshold=0.95, device=None):
    """
    Pseudo-labeling strategy to use unlabeled data

    Args:
        model (callable): Pre-trained model
        unlabeled_images (np.ndarray): Images without labels
        channel_means (iterable): Channel means for normalization
        channel_stds (iterable): Channel standard deviations for normalization
        confidence_threshold (float): Confidence threshold to accept a pseudo-label
        device (torch.device): Device to run the model on (CPU or GPU)

    Returns:
        pseudo_labeled_images, pseudo_labels: High-confidence pseudo-labeled data
    """

    if device is None:
        device = get_device()

    model.eval()
    transform = create_transforms(channel_means, channel_stds)
    unlabeled_dataset = CustomImageDataset(unlabeled_images, labels=None, transform=transform)
    batch_size = 16
    loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=2)

    pseudo_images = []
    pseudo_labels = []
    confidences = []

    print(f"Pseudo-label generation with confidence threshold: {confidence_threshold}")
    print(f"Processing {len(unlabeled_images)} unlabeled images...")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            # Move images to device
            images = images.to(device)

            # Prediction
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            # Check confidence for each image in the batch
            for image_idx, (max_prob, pred_class) in enumerate(zip(max_probs, predicted)):
                if max_prob.item() >= confidence_threshold:
                    # Get the original image (before transformation) to maintain consistent format
                    original_idx = batch_idx * batch_size + image_idx
                    if original_idx < len(unlabeled_images):
                        original_image = unlabeled_images[original_idx]
                        pseudo_images.append(original_image)
                        
                        class_names = ["NO_NUCLEUS", "NUCLEI"]
                        pseudo_labels.append(class_names[pred_class.item()])
                        confidences.append(max_prob.item())

            # Progress update every 20 batches
            if (batch_idx + 1) % 20 == 0:
                processed = min((batch_idx + 1) * batch_size, len(unlabeled_images))
                print(f"Processed {processed}/{len(unlabeled_images)} images")

    print(f"Pseudo-labels generated: {len(pseudo_images)}/{len(unlabeled_images)} "
          f"({100*len(pseudo_images)/len(unlabeled_images):.1f}%)")

    if len(pseudo_images) > 0:
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Distribution of pseudo-labels: {np.unique(pseudo_labels, return_counts=True)}")

    return np.array(pseudo_images), np.array(pseudo_labels)


def train_with_pseudo_labeling(train_images, train_labels, test_images, test_labels,
                              unlabeled_images, channel_means, channel_stds,
                              val_images=None, val_labels=None, initial_model_path=None,
                              n_iterations=3, num_epochs=50, batch_size=16,
                              learning_rate=0.001, weight_decay=1e-4, confidence_threshold=0.95,
                              device=None, verbose=1, **kwargs):
    """
    Training with pseudo-labeling strategy
    
    Args:
        train_images: Labeled training images
        train_labels: Labeled training labels
        test_images: Test images
        test_labels: Test labels
        unlabeled_images: Images without labels to use for pseudo-labeling
        channel_means: Channel means for normalization
        channel_stds: Channel standard deviations for normalization
        val_images: Validation images (optional)
        val_labels: Validation labels (optional)
        initial_model_path: Path to an initial pre-trained model (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for L2 regularization
        confidence_threshold: Confidence threshold for pseudo-labels
        device: Device to run the model on (CPU or GPU)
        verbose: Verbosity level (0: silent, 1: progress bar, 2: detailed logging)
        kwargs: Additional arguments for training (e.g., dropout rate, use_early_stopping)
    """

    if verbose > 0:
        print("=" * 80)
        print("TRAINING WITH PSEUDO-LABELING")
        print("=" * 80)

    if device is None:
        device = get_device()

    # Phase 1: Initial training with labeled data
    if verbose > 0:
        print("\n🚀 PHASE 1: Initial training with labeled data")
        print("-" * 60)

    if initial_model_path is not None:
        print(kwargs)
        initial_model = load_model(initial_model_path)[0]
        if verbose > 1:
            print(f"Loaded initial model from {initial_model_path}")
    else:
        initial_model, _ = train_model_with_regularization(
            train_images, train_labels, test_images, test_labels,
            channel_means, channel_stds, val_images=val_images, val_labels=val_labels,
            num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, device=device,
            use_early_stopping=True, verbose=verbose, **kwargs
        )
        if verbose > 1:
            print("Initial model trained with labeled data")

    # Variables for iterative training
    current_train_images = train_images.copy()
    current_train_labels = train_labels.copy()
    all_metrics = []

    if verbose > 0:
        print(f"\n🔄 PHASE 2: Iterative training with pseudo-labeling")
        print("-" * 60)

    for iteration in range(n_iterations):
        if verbose > 0:
            print(f"\n--- ITERATION {iteration + 1}/{n_iterations} ---")

        # Generate pseudo-labels
        pseudo_images, pseudo_labels = pseudo_labeling_strategy(
            initial_model, unlabeled_images, channel_means, channel_stds,
            confidence_threshold=confidence_threshold, device=device
        )

        if len(pseudo_images) == 0:
            if verbose > 0:
                print("❌ No pseudo-label generated, stopping pseudo-labeling")
            break

        # Add pseudo-labels to training data
        augmented_train_images = np.concatenate([current_train_images, pseudo_images], axis=0)
        augmented_train_labels = np.concatenate([current_train_labels, pseudo_labels], axis=0)
        
        if verbose > 0:
            print(f"📈 Augmented dataset: {len(current_train_images)} → "
                f"{len(augmented_train_images)} images")

        # Retrain with augmented data
        model, metrics = train_model_with_regularization(
            augmented_train_images, augmented_train_labels, test_images, test_labels,
            channel_means, channel_stds, val_images=val_images, val_labels=val_labels,
            num_epochs=num_epochs, batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay, device=device,
            use_early_stopping=True, verbose=verbose,
            label_smoothing=0.1, **kwargs
        )

        all_metrics.append(metrics["final_metrics"])
        initial_model = model  # Use the new model

        # Optional: Increase the confidence threshold for the next iteration
        # confidence_threshold = min(0.98, confidence_threshold + 0.01)

    print("\n✅ Pseudo-labeling completed!")

    # Performance comparison
    print("\n📊 PERFORMANCE EVOLUTION:")
    print("-" * 40)
    for i, metrics in enumerate(all_metrics):
        print(f"Iteration {i+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    return model, all_metrics


def comprehensive_training_strategy(data_path, validation=False,
                                   strategy="regularization", num_epochs=30, 
                                   batch_size=16, learning_rate=0.001, **kwargs):
    """
    Training strategy for restricted datasets with overfitting

    Args:
        data_path: Path to the labeled data
        unlabeled_data_path: Path to the unlabeled data (optional)
        strategy: 'regularization' or 'pseudo_labeling'

    Available anti-overfitting strategies:
    1. 'regularization': Data augmentation + early stopping + L2 regularization
    2. 'pseudo_labeling': Use of unlabeled data
    """

    print("=" * 80)
    print("Complete anti-overfitting strategy")
    print("=" * 80)
    print(f"Selected strategy: {strategy}")

    # Load data
    data = load_data(data_path, validation=validation)
    train_images = data["train_images"]
    train_labels = data["train_labels"]
    test_images = data["test_images"] 
    test_labels = data["test_labels"]
    val_images = data.get("val_images", None)
    val_labels = data.get("val_labels", None)
    channel_means = data["channel_means"]
    channel_stds = data["channel_stds"]
    labeled_images = data["labeled_images"]
    unlabeled_images = data["unlabeled_images"]

    print(f"📊 Loaded data:")
    print(f"   - Train: {len(train_images)} images")
    print(f"   - Test: {len(test_images)} images")
    print(f"   - Total labeled: {len(labeled_images)} images")
    print(f"   - Total unlabeled: {len(unlabeled_images)} images")

    # Execute the selected strategy
    if strategy == "regularization":
        print("\n🛡️ STRATEGY: Advanced Regularization")
        model, history = train_model_with_regularization(
            train_images, train_labels, test_images, test_labels,
            channel_means, channel_stds, val_images=val_images,
            val_labels=val_labels, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            **kwargs
        )
    elif strategy == "pseudo_labeling":
        print("\n🏷️ STRATEGY: Pseudo-labeling")
        model, history = train_with_pseudo_labeling(
            train_images, train_labels, test_images, test_labels,
            unlabeled_images, channel_means, channel_stds,
            val_images=val_images, val_labels=val_labels,
            num_epochs=num_epochs, batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
    else:
        print("❌ Strategy not recognized")
        print("   Using default regularization strategy")
        model, history = train_model_with_regularization(
            train_images, train_labels, test_images, test_labels,
            channel_means, channel_stds, val_images=val_images,
            val_labels=val_labels, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            **kwargs
        )

    # Save the final model
    model_path = f"models/nuclei_classifier_{strategy}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "history": history,
        "strategy": strategy
    }, model_path)

    print(f"\n💾 Model saved: {model_path}")

    return model, history
