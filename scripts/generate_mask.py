import os
import torch
import tempfile
import openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from nuclei_detection.loader import load_model
from nuclei_detection.dataset import create_transforms


def generated_nuclei_mask(wsi_path, model_path, patch_size=256, stride=128, batch_size=32, temp_dir=None):
    """
    Generate a mask for nuclei in a whole slide image (WSI) using a pre-trained model.
    Uses memory-mapped arrays to handle large WSI images efficiently.

    Args:
        wsi_path (str): Path to the whole slide image.
        patch_size (int): Size of the patches to extract from the WSI.
        stride (int): Stride for sliding window extraction.
        batch_size (int): Number of patches to process in parallel.
        temp_dir (str): Directory for temporary files. If None, uses system temp directory.

    Returns:
        np.ndarray: Binary mask indicating the presence of nuclei.
    """
    # Load the WSI and extract patches        
    wsi = openslide.open_slide(wsi_path)
    width, height = wsi.dimensions

    print(f"WSI dimensions: {width} x {height}")
    print(f"Estimated memory usage without disk storage: "
          f"{(width * height * 2 * 4 + width * height * 4) / 1024**3:.2f} GB")

    # Create temporary directory for memory-mapped arrays
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="wsi_processing_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    # Create memory-mapped arrays stored on disk
    prob_accumulation_path = os.path.join(temp_dir, "prob_accumulation.dat")
    count_matrix_path = os.path.join(temp_dir, "count_matrix.dat")

    # Initialize memory-mapped arrays
    prob_accumulation = np.memmap(
        prob_accumulation_path,
        dtype=np.float32,
        mode="w+",
        shape=(height, width, 2)
    )
    count_matrix = np.memmap(
        count_matrix_path,
        dtype=np.uint8,
        mode="w+",
        shape=(height, width)
    )

    # Initialize arrays to zero
    prob_accumulation[:] = 0
    count_matrix[:] = 0

    # Force write to disk
    prob_accumulation.flush()
    count_matrix.flush()

    # Load the pre-trained model
    model, channel_means, channel_stds, device, _ = load_model(model_path)
    model.eval()

    transform = create_transforms(channel_means, channel_stds)

    # Calculate patch positions ensuring complete coverage
    def calculate_patch_positions(total_size, patch_size, stride):
        """Calculate patch positions ensuring complete coverage of the image"""
        positions = []
        pos = 0
        while pos <= total_size - patch_size:
            positions.append(pos)
            pos += stride

        # Ensure the last patch covers the end of the image
        if positions[-1] + patch_size < total_size:
            positions.append(total_size - patch_size)

        return positions

    y_positions = calculate_patch_positions(height, patch_size, stride)
    x_positions = calculate_patch_positions(width, patch_size, stride)

    # Create all patch coordinates
    patch_coords = [(y, x) for y in y_positions for x in x_positions]
    total_patches = len(patch_coords)

    print(f"Processing {total_patches} patches with stride={stride}, patch_size={patch_size}, batch_size={batch_size}")
    print(f"Y positions: {len(y_positions)}, X positions: {len(x_positions)}")
    print(f"Image size: {width}x{height}")

    # Process patches in batches
    for batch_start in range(0, total_patches, batch_size):
        batch_end = min(batch_start + batch_size, total_patches)
        batch_coords = patch_coords[batch_start:batch_end]

        # Extract patches for this batch
        batch_patches = []
        batch_positions = []

        for y, x in batch_coords:
            patch = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            batch_patches.append(patch)
            batch_positions.append((y, x))

        # Convert patches to tensors
        batch_tensors = []
        for patch in batch_patches:
            image_tensor = transform(patch)
            batch_tensors.append(image_tensor)

        # Stack into a batch tensor
        batch_tensor = torch.stack(batch_tensors).to(device)  # Shape: (batch_size, C, H, W)

        # Make batched prediction
        with torch.no_grad():
            outputs = model(batch_tensor)  # Shape: (batch_size, num_classes)
            probabilities = torch.softmax(outputs, dim=1)  # Shape: (batch_size, 2)

        # Convert probabilities to numpy
        probs_np = probabilities.cpu().numpy()  # Shape: (batch_size, 2)

        # Accumulate probabilities for each patch in the batch
        for i, (y, x) in enumerate(batch_positions):
            # Use slicing to update memory-mapped arrays efficiently
            y_end = y + patch_size
            x_end = x + patch_size

            prob_accumulation[y:y_end, x:x_end, 0] += probs_np[i, 0]  # NO_NUCLEUS
            prob_accumulation[y:y_end, x:x_end, 1] += probs_np[i, 1]  # NUCLEI
            count_matrix[y:y_end, x:x_end] += 1

        # Flush to disk periodically to ensure data is written
        if (batch_start // batch_size + 1) % 50 == 0:
            prob_accumulation.flush()
            count_matrix.flush()

        # Progress reporting
        if (batch_start // batch_size + 1) % 10 == 0 or batch_end == total_patches:
            print(f"Processed {batch_end}/{total_patches} patches ({100*batch_end/total_patches:.1f}%)")

    # Final flush to ensure all data is written
    prob_accumulation.flush()
    count_matrix.flush()

    # Average the accumulated probabilities
    print("Computing final predictions...")

    # Process in chunks to avoid loading entire arrays into memory
    chunk_size = 1000  # Process 1000 rows at a time

    # Create memory-mapped array for final mask
    mask_path = os.path.join(temp_dir, "final_mask.dat")
    final_mask = np.memmap(
        mask_path,
        dtype=np.uint8,
        mode="w+",
        shape=(height, width)
    )

    print("Processing mask in chunks to save memory...")
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)

        # Load chunk into memory
        chunk_probs = prob_accumulation[start_row:end_row, :, :].copy()
        chunk_counts = count_matrix[start_row:end_row, :].copy()

        # Average probabilities
        chunk_probs = chunk_probs / chunk_counts[:, :, np.newaxis]

        # Apply argmax to get final predictions
        chunk_mask = np.argmax(chunk_probs, axis=-1).astype(np.uint8)

        # Store result
        final_mask[start_row:end_row, :] = chunk_mask

        if (start_row // chunk_size + 1) % 10 == 0:
            progress = min(100, 100 * end_row / height)
            print(f"Processed {progress:.1f}% of final mask computation")

    final_mask.flush()

    # Convert memory-mapped array to regular array for the final result
    mask = np.array(final_mask)

    print(f"Final mask shape: {mask.shape}")
    print(f"Unique values in mask: {np.unique(mask, return_counts=True)}")

    # Convert to binary image (0->0, 1->255) for visualization
    mask_image = Image.fromarray(mask * 255)

    wsi_extension = wsi_path.split(".")[-1]
    mask_filename = wsi_path.replace(f".{wsi_extension}", f"_nuclei_mask.png")
    mask_image.save(mask_filename)
    print(f"Nuclei mask saved as: {mask_filename}")

    # Cleanup temporary files
    try:
        print("Cleaning up temporary files...")
        os.unlink(prob_accumulation_path)
        os.unlink(count_matrix_path)
        os.unlink(mask_path)
        os.rmdir(temp_dir)
        print("Temporary files cleaned up successfully")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")
        print(f"Temporary directory: {temp_dir}")

    return mask


def display_wsi_with_mask(wsi_path, mask_path=None, level_downsample=4, save=False):
    """
    Display the whole slide image alongside its generated nuclei mask at a specified zoom level.
    Automatically calculates optimal figure size based on WSI dimensions.

    Args:
        wsi_path (str): Path to the whole slide image.
        mask_path (str): Path to the generated nuclei mask (binary mask with values 0 and 1).
        level_downsample (int): Zoom level for display (0 = highest resolution, higher = more downsampled).
        save (bool): Whether to save the displayed image to disk.
    """

    # Open the WSI
    wsi = openslide.open_slide(wsi_path)
    print(f"WSI level dimensions: {wsi.level_dimensions}")
    print(f"WSI level downsamples: {wsi.level_downsamples}")

    # Get dimensions at the specified zoom level
    if level_downsample >= len(wsi.level_dimensions):
        level_downsample = len(wsi.level_dimensions) - 1
        print(f"Requested zoom level not available, using level {level_downsample}")

    downsample_factor = wsi.level_downsamples[level_downsample]
    level_width, level_height = wsi.level_dimensions[level_downsample]

    print(f"Display level {level_downsample}: {level_width} x {level_height}")
    print(f"Downsample factor: {downsample_factor:.2f}")

    # Calculate optimal figure size based on WSI aspect ratio
    # We have 3 subplots side by side, so total width = 3 * individual_width
    aspect_ratio = level_width / level_height

    # Reduced base height for more compact display
    base_height = 4  # Reduced from 6 to 4 for more compact display

    # Calculate width based on aspect ratio, accounting for 3 subplots
    individual_subplot_width = base_height * aspect_ratio
    total_width = individual_subplot_width * 3

    # Ensure reasonable bounds (smaller than before)
    total_width = max(10, min(18, total_width))  # Between 10 and 18 inches wide (reduced from 12-24)
    figure_height = max(4, min(8, base_height))  # Between 4 and 8 inches tall (reduced from 6-12)

    # Add minimal extra height for title and statistics
    figure_height += 1.5  # Reduced from 2 to 1.5 for less whitespace

    figsize = (total_width, figure_height)
    print(f"Calculated figure size: {figsize[0]:.1f} x {figsize[1]:.1f} inches")

    # Read the WSI at the specified level
    wsi_image = wsi.read_region((0, 0), level_downsample, (level_width, level_height))
    wsi_image = wsi_image.convert("RGB")
    wsi_array = np.array(wsi_image)

    if mask_path is None:
        extension = wsi_path.split(".")[-1]
        mask_path = wsi_path.replace(f".{extension}", "_nuclei_mask.png")
    # Load the mask as a regular PIL image and downsample it to match WSI level
    print(f"Loading mask from: {mask_path}")

    # Set PIL to allow large images
    Image.MAX_IMAGE_PIXELS = None

    # Load the full resolution mask
    mask_image = Image.open(mask_path).convert("L")
    mask_width, mask_height = mask_image.size
    print(f"Original mask size: {mask_width} x {mask_height}")

    # Downsample the mask to match the WSI level
    target_width = int(mask_width / downsample_factor)
    target_height = int(mask_height / downsample_factor)
    print(f"Downsampling mask to: {target_width} x {target_height}")

    # Use LANCZOS for high-quality downsampling of the mask
    mask_image = mask_image.resize((target_width, target_height), Image.LANCZOS)
    mask_array = np.array(mask_image) / 255.0

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Display original WSI
    axes[0].imshow(wsi_array)
    axes[0].set_title(f"Original WSI (Level {level_downsample})", fontsize=14)
    axes[0].axis("off")

    # Display mask
    axes[1].imshow(mask_array, cmap="gray")
    axes[1].set_title("Generated Nuclei Mask", fontsize=14)
    axes[1].axis("off")

    # Display overlay
    # Create a colored mask overlay (red for nuclei)
    overlay = wsi_array.copy().astype(np.float32)
    nuclei_mask = mask_array > 0.5
    overlay[nuclei_mask, 0] = np.minimum(255, overlay[nuclei_mask, 0] * 0.6 + 255 * 0.4)  # Add red
    overlay[nuclei_mask, 1] = overlay[nuclei_mask, 1] * 0.6  # Reduce green
    overlay[nuclei_mask, 2] = overlay[nuclei_mask, 2] * 0.6  # Reduce blue

    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("WSI with Nuclei Overlay (Red)", fontsize=14)
    axes[2].axis("off")

    # Add overall title
    wsi_name = os.path.basename(wsi_path)
    fig.suptitle(f"Nuclei Detection Results: {wsi_name}", fontsize=14, y=0.92)  # Reduced fontsize and y position

    # Add statistics
    total_pixels = mask_array.size
    nuclei_pixels = np.sum(mask_array > 0.5)
    nuclei_percentage = (nuclei_pixels / total_pixels) * 100

    fig.text(0.5, 0.05, f"Nuclei Coverage: {nuclei_percentage:.2f}% ({nuclei_pixels:,} / {total_pixels:,} pixels)", 
             ha="center", fontsize=11)  # Reduced fontsize and moved up slightly

    # Optimize layout to minimize whitespace
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.95)  # Tighter margins

    # Save the figure if requested
    if save:
        wsi_extension = wsi_path.split(".")[-1]
        save_filename = wsi_path.replace(f".{wsi_extension}", f"_with_mask.png")
        plt.savefig(save_filename, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"Visualization saved as: {save_filename}")

    plt.show()
    
    # Close WSI
    wsi.close()
