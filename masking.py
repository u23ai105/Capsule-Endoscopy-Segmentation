import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

def generate_and_save_masks(root_path, split_name, annotation_fname, output_mask_dir):
    """
    root_path: Root directory containing images and annotation
    split_name: e.g. "train" or "test"
    annotation_fname: COCO-style annotation file (e.g., "annotations.json")
    output_mask_dir: Directory where the generated masks will be saved
    """
    # Setup paths
    annotation_path = os.path.join(root_path, split_name, annotation_fname)
    image_dir = os.path.join(root_path, split_name)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Load COCO annotations
    coco = COCO(annotation_path)
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc="Generating masks"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        height, width = img_info["height"], img_info["width"]

        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Load and combine annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            mask |= coco.annToMask(ann)

        # Save mask as PNG
        base_name = os.path.splitext(file_name)[0]
        mask_path = os.path.join(output_mask_dir, base_name + ".png")
        cv2.imwrite(mask_path, mask * 255)

# Example usage
generate_and_save_masks(
    root_path="/content/brain-tumor-image-dataset-semantic-segmentation",
    split_name="valid",
    annotation_fname="_annotations.coco.json",
    output_mask_dir="/content/brain-tumor-image-dataset-semantic-segmentation/valid_masks"
)

generate_and_save_masks(
    root_path="/content/brain-tumor-image-dataset-semantic-segmentation",
    split_name="test",
    annotation_fname="_annotations.coco.json",
    output_mask_dir="/content/brain-tumor-image-dataset-semantic-segmentation/test_masks"
)

generate_and_save_masks(
    root_path="/content/brain-tumor-image-dataset-semantic-segmentation",
    split_name="train",
    annotation_fname="_annotations.coco.json",
    output_mask_dir="/content/brain-tumor-image-dataset-semantic-segmentation/train_masks"
)
