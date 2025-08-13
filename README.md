# Nuclei Detection and Mask Generation

This project implements a machine learning pipeline for detecting cell nuclei in histopathology images and generating binary masks highlighting nuclei-containing regions.

## Problem Overview

Given a dataset of 256x256 histopathology image tiles (some labeled as NUCLEI/NO_NUCLEUS), the project solves two key challenges:

1. **Classification**: Train a robust binary classifier to identify tiles containing cell nuclei
2. **Mask Generation**: Apply the trained model to whole slide images (WSI) or large crops to generate binary masks where white pixels indicate nuclei presence

## Approach Summary

**Model Architecture**: Adapted LeNet-5 CNN for 256x256 RGB images with dropout regularization

**Training Strategy**: Multi-stage approach combining:
- Regularized training with data augmentation, early stopping, and learning rate scheduling
- Pseudo-labeling to leverage unlabeled data (semi-supervised learning)
- Iterative refinement through multiple pseudo-labeling cycles

**Mask Generation**: Sliding window approach with overlapping patches and probability aggregation for robust predictions

## Projet structure

The project follows a classical `uv` project style:
```
nuclei_detection/
├── main.py                     # Main entry point with CLI
├── pyproject.toml              # Project dependencies and configuration
├── README.md                   # This documentation
├── data/                       # Dataset storage
├── models/                     # Saved model checkpoints
├── nuclei_detection/           # Core Python utilities
│   ├── loader.py              # Data and model loading
│   ├── dataset.py             # Torch dataset and augmentation
│   ├── model.py               # LeNet-5 architecture implementation
│   ├── training.py            # Training loops and strategies
│   ├── evaluation.py          # Model evaluation utilities
│   └── utils.py               # Helper functions
└── scripts/                    # Individual script modules
    ├── explore_dataset.py     # Dataset analysis and visualization
    ├── train_model.py         # Training implementations
    ├── evaluate_model.py      # Model evaluation
    └── generate_mask.py       # Mask generation for WSI
```

It additionally uses the `fire` package which directly turns functions into scripts (function arguments being passed as script arguments).

## Quick Start

### Prerequisites
- `uv` package manager (see this [webpage](https://docs.astral.sh/uv/getting-started/installation/) for installation)

### Installation
```bash
# Clone the repository and navigate to the project directory
cd nuclei_detection

# Install all dependencies
uv sync
```

### Data Setup
1. Download the required datasets:
   - `dataset_nuclei_tiles.zip` (training data with labels)
   - `wsi_crops.zip` (test images for mask generation)
   - `dataset_WSI.zip` (two lightest WSI for mask generation)
2. Extract both into the `data/` folder
3. Your data structure should look like:
   ```
   data/
   ├── dataset_nuclei_tiles/
   │   ├── labels.csv
   │   └── tile_*.png
   └── wsi_crops/
   │    └── wsi_crop_*.jpg
   └── dataset_WSI/... (optional)
   ```

### Usage Examples

#### 1. Train the Nuclei Classifier
```bash
# Quick training (basic setup)
uv run main.py train-model ./data/dataset_nuclei_tiles

# Recommended training with regularization and pseudo-labeling
uv run main.py train-model-with-regularization ./data/dataset_nuclei_tiles \
    --strategy pseudo_labeling --use-cutout --patience 5 --validation
```

#### 2. Evaluate Model Performance
```bash
# Evaluate on test set
uv run main.py evaluate-model ./data/dataset_nuclei_tiles \
    --model-path ./models/nuclei_classifier_pseudo_labeling.pth
```

#### 3. Generate Binary Masks
```bash
# Generate mask for a WSI crop
uv run main.py generate-nuclei-mask \
    --wsi-path ./data/wsi_crops/wsi_crop_1.jpg \
    --model-path ./models/nuclei_classifier_pseudo_labeling.pth

# Display results with overlay visualization
uv run main.py display-wsi-with-mask \
    --wsi-path ./data/wsi_crops/wsi_crop_1.jpg
```

## Technical Approach

### 1. Problem Analysis and Data Challenges

**Dataset Characteristics:**
- Small dataset (<5000 images, only ~2000 labeled)
- Class imbalance between NUCLEI/NO_NUCLEUS samples

**Key Challenges Identified:**
- Overfitting due to small dataset size
- Effective utilization of unlabeled data
- Generalization to different staining and imaging conditions

### 2. Model Architecture

**Adapted LeNet-5 for Medical Imaging:**

```
Input: (3, 256, 256) RGB histopathology tiles
│
├── Conv2d(3→6, kernel=5, padding=2) + ReLU
├── AvgPool2d(kernel=2, stride=2)
│
├── Conv2d(6→16, kernel=5) + ReLU  
├── AvgPool2d(kernel=2, stride=2)
│
├── Conv2d(16→120, kernel=5) + ReLU
│
├── Flatten
├── Linear(120×58×58 → 84) + ReLU + Dropout(0.5)
├── Linear(84 → 2)
│
Output: (2,) Logits for [NO_NUCLEUS, NUCLEI]
```

**Design Rationale:**
- LeNet-5 chosen for its simplicity and effectiveness on smaller datasets
- Dropout layer added to prevent overfitting
- Architecture adapted to handle 256x256 RGB input instead of original 32x32 grayscale

### 3. Training Methodology

**Multi-Stage Training Pipeline:**

1. **Initial Supervised Training**
   - Train on labeled data with heavy regularization
   - Data augmentation: geometric transforms, color jitter, cutout
   - Early stopping with validation monitoring
   - Label smoothing to provide better calibrated models with less confidence

2. **Pseudo-Labeling (Semi-Supervised Learning)**
   - Use trained model to label unlabeled tiles
   - Confidence thresholding to ensure quality pseudo-labels
   - Iterative refinement over multiple cycles

3. **Regularization Techniques**
   - Weight decay (L2 regularization): 1e-4
   - Gradient clipping for training stability
   - Learning rate scheduling (ReduceLROnPlateau)
   - Dropout: 0.5 in fully connected layers

### 4. Mask Generation Strategy

**Sliding Window with Overlap:**
- Extract 256x256 patches with configurable stride (default: 128px)
- Multiple predictions per pixel for robustness
- Probability aggregation across overlapping predictions

**Memory-Efficient Processing:**
- NumPy memory mapping for large WSI processing
- Temporary disk storage for probability maps
- Automatic cleanup after mask generation

**Output:**
- Binary masks saved as PNG images
- White pixels indicate nuclei presence
- Visualization tools for overlay analysis

## Performance and Results

### Model Evaluation Metrics
The training pipeline monitors comprehensive metrics:
- **Cross-Entropy Loss**
- **Accuracy**
- **Precision, Recall, F1-Score** (weighted averages)
- **Confusion Matrix**

### Best Model Performance
**Final Test Results (Pseudo-Labeling Strategy):**
- **Test F1-Score: 97.33%**
- Training command used:
```bash
uv run main.py train-model-with-regularization ./data/dataset_nuclei_tiles \
    --strategy pseudo_labeling --use-cutout --patience 5 --validation --label-smoothing 0.1
```

### Training Visualization
With `verbose=2`, the training generates diagnostic plots:
- Learning curves (Loss & Accuracy over epochs)
- Learning rate evolution
- Train/validation gap analysis (overfitting detection)
- Final confusion matrix

## Additional Features

### Dataset Exploration
```bash
# Analyze dataset statistics and visualizations
uv run main.py explore-dataset ./data/dataset_nuclei_tiles
```
- Class distribution analysis
- Sample visualization
- Data quality assessment

### Full WSI Processing (Bonus)
For complete whole slide images (requires unzipped `dataset_WSI.zip`):
```bash
# Process full WSI (disk memory-intensive)
uv run main.py generate-nuclei-mask \
    --wsi-path ./data/dataset_WSI/H0709980/H0709980-01.ndpi \
    --model-path ./models/nuclei_classifier_pseudo_labeling.pth

# Visualize it along with its produced mask
uv run main.py display-wsi-with-mask \
    --wsi-path ./data/dataset_WSI/H0709980/H0709980-01.ndpi
```

**Technical Implementation:**
- Leverages OpenSlide library for WSI format support
- Memory-mapped arrays for handling large probability maps
- Configurable tile stride for speed/accuracy trade-off


### Deliverables

1. **Trained Model**: `models/nuclei_classifier_pseudo_labeling.pth`
   - Binary classifier for NUCLEI/NO_NUCLEUS detection

2. **Generated Binary Masks**: Saved as PNG files. All wsi crops masks and one WSI mask are available
   - Example: mask for `wsi_crop_1.jpg` named `wsi_crop_1_nuclei_mask.png`
   - Overlaid visualizations available via display commands and a sample for the WSI is provided


## Future Improvements

### Model Robustness
- **Blur Tolerance**: Add GaussianBlur augmentation for handling noisy/blurry images (e.g. wsi_crop_5.jpg is blurry and the output mask seem off)
- **Model Ensemble**: Combine 3-5 models for improved accuracy and robustness
- **Transfer Learning**: Fine-tune pre-trained histopathology models (e.g., from TIA Toolbox which contains nuclei detection pretrained models and more)

### Mask Generation Optimization
- **Model Performance** The current F1-score of 97.33% may be insufficient for a robust preprocessing step, as it is applied many time on a given WSI and still makes mistakes
- **Tissue Segmentation**: Pre-filter tissue regions to reduce computational and memory overhead
- **TIA Toolbox Integration**: Leverage specialized WSI processing functions
