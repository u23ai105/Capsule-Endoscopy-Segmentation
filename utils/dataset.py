import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
from PIL import Image
import glob


class SemanticSegmentationDataset(Dataset):
    """
    A robust Dataset class for semantic segmentation.
    This version corrects file discovery and uses a standard torchvision transform pipeline.
    """

    def __init__(self, img_dir, mask_dir, mode='binary', image_transforms=None, mask_transforms=None):
        """
        Parameters
        ----------
        img_dir : str
            Path to the directory containing images.
        mask_dir : str
            Path to the directory containing masks.
        mode : str, default='binary'
            'binary' for binary segmentation.
        image_transforms : torchvision.transforms.Compose, default=None
            A composition of transforms to apply to the images.
        mask_transforms : torchvision.transforms.Compose, default=None
            A composition of transforms to apply to the masks.
        """
        # ERROR FIX: The original _get_file_dir was buggy.
        # This uses glob, which is simpler and more reliable for finding all files.
        self.img_paths = self._get_file_dir(img_dir)
        self.mask_paths = self._get_file_dir(mask_dir)

        self.mode = mode
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        # Ensure the number of images and masks match
        assert len(self.img_paths) > 0, f"No images found in {img_dir}"
        assert len(self.img_paths) == len(self.mask_paths), \
            f"Mismatch between number of images ({len(self.img_paths)}) and masks ({len(self.mask_paths)})."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Loads and returns a sample from the dataset at the given index.
        """
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale

        if self.image_transforms:
            img = self.image_transforms(img)

        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        # Binarize the mask after transformations
        if self.mode == 'binary':
            mask[mask > 0] = 1.0

        return img, mask

    def _get_file_dir(self, directory):
        """
        Correctly returns all image files in the entered directory and its subdirectories,
        ignoring non-image files.
        """

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            # Sorts files in a human-friendly order (e.g., 1, 2, 10 instead of 1, 10, 2)
            return [atoi(c) for c in re.split('(\d+)', text)]

        all_files = []
        # Define the extensions to look for
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

        if os.path.exists(directory):
            # Walk through all directories and subdirectories
            for root, _, files in os.walk(directory):
                for file in files:
                    # Check if the file has one of the specified image extensions
                    if file.lower().endswith(image_extensions):
                        all_files.append(os.path.join(root, file))

        # Sort the final list to ensure alignment between images and masks
        all_files.sort(key=natural_keys)
        return all_files