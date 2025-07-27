import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
from google.colab import drive


def display_segmentation_results():
    """
    Loads original images, ground truth masks, and saved prediction masks,
    then creates and saves side-by-side comparison plots.
    """
    # --- 1. Mount Google Drive ---
    print("Mounting Google Drive...")
    try:
        drive.mount('/content/gdrive', force_remount=True)
        print("Google Drive mounted successfully.")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        return

    # --- 2. Configuration: DEFINE YOUR FOLDER PATHS HERE ---

    # Path to the folder with your original test images
    original_images_path = '/content/brain-tumor-image-dataset-semantic-segmentation/test'

    # Path to the folder with the ground truth masks
    ground_truth_masks_path = '/content/brain-tumor-image-dataset-semantic-segmentation/test_masks'

    # Path to the folder where your model's predictions were saved
    # NOTE: Please double-check this path is correct.
    predicted_masks_path = '/content/gdrive/My Drive/segmentation_model/test_predictions'

    # Path in your Google Drive to save the final comparison plots
    output_save_path = '/content/gdrive/My Drive/segmentation_model/final_comparisons'

    # Number of random samples you want to visualize
    num_samples_to_display = 10

    # --- 3. Setup and Validation ---
    os.makedirs(output_save_path, exist_ok=True)

    for path in [original_images_path, ground_truth_masks_path, predicted_masks_path]:
        if not os.path.exists(path):
            print(f"Error: The specified path does not exist: {path}")
            return

    # --- 4. Find and Match Files ---
    # Get a list of all original image files
    image_files = sorted(glob.glob(os.path.join(original_images_path, '*.*')))

    if not image_files:
        print(f"Error: No images found in '{original_images_path}'")
        return

    # Select random samples to display
    if len(image_files) > num_samples_to_display:
        image_files_to_process = np.random.choice(image_files, num_samples_to_display, replace=False)
    else:
        image_files_to_process = image_files
        print(
            f"Warning: Found fewer images ({len(image_files)}) than requested samples ({num_samples_to_display}). Processing all found images.")

    print(f"Generating {len(image_files_to_process)} comparison plots...")

    # --- 5. Generate and Save Plots ---
    for img_path in image_files_to_process:
        try:
            filename_with_ext = os.path.basename(img_path)
            # **FIX**: Get the filename without the extension
            base_filename = os.path.splitext(filename_with_ext)[0]

            # **FIX**: Construct the mask path with the correct .png extension
            gt_mask_path = os.path.join(ground_truth_masks_path, base_filename + '.png')
            # The prediction was likely saved with the original full filename
            pred_mask_path = os.path.join(predicted_masks_path, filename_with_ext)

            # Check if both corresponding masks exist
            if not os.path.exists(gt_mask_path):
                print(f"Skipping {filename_with_ext}: Ground truth mask not found at {gt_mask_path}")
                continue
            if not os.path.exists(pred_mask_path):
                print(f"Skipping {filename_with_ext}: Predicted mask not found at {pred_mask_path}")
                continue

            # Load the three images
            original_image = Image.open(img_path).convert("RGB")
            gt_mask = Image.open(gt_mask_path).convert("L")
            pred_mask = Image.open(pred_mask_path).convert("L")

            # Create the plot
            fig, ax = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'Comparison for: {filename_with_ext}', fontsize=16)

            ax[0].imshow(original_image)
            ax[0].set_title("Original Image")
            ax[0].axis('off')

            ax[1].imshow(gt_mask, cmap='gray')
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis('off')

            ax[2].imshow(pred_mask, cmap='gray')
            ax[2].set_title("Predicted Mask")
            ax[2].axis('off')

            # Save the figure to the specified output path in Google Drive
            output_filename = f"comparison_{filename_with_ext}"
            full_save_path = os.path.join(output_save_path, output_filename)
            plt.savefig(full_save_path)
            plt.close(fig)  # Close the figure to free up memory

        except Exception as e:
            print(f"Could not process {filename_with_ext}. Error: {e}")

    print("\n--- Visualization Complete! ---")
    print(f"Comparison plots have been saved to: {output_save_path}")


if __name__ == '__main__':
    display_segmentation_results()
