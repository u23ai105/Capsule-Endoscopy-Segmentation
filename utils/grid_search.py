import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import itertools
import csv
import torchvision.transforms as T

# --- Add /content to path and import your modules ---
if '/content' not in sys.path:
    sys.path.append('/content')

from custom_unet import Unet
from dataset import SemanticSegmentationDataset
from losses import DiceLoss,DiceBCELoss,FocalLoss
from reproducibility import set_seed
from trainer import Trainer


def run_training(params, base_save_path):
    """
    A wrapper function to run a single training instance with a given set of parameters.
    """
    # --- Configuration ---
    set_seed(42)

    # --- Unpack parameters from the grid ---
    backbone = params['backbone']
    img_size = params['img_size']
    batch_size = params['batch_size']
    lr = params['lr']
    loss_name = params['loss_function']

    print("\n" + "=" * 50)
    print(f"Starting Run with Params: {params}")
    print("=" * 50 + "\n")

    # --- Create a unique, structured directory for this run IN GOOGLE DRIVE ---
    # Main directory for the backbone
    backbone_dir = os.path.join(base_save_path, backbone)
    # Specific subdirectory for this hyperparameter combination
    run_name = f"loss_{loss_name}_lr_{lr}_size_{img_size}_bs_{batch_size}"
    save_dir = os.path.join(backbone_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Paths ---
    train_img_path = '/content/brain-tumor-image-dataset-semantic-segmentation/train'
    train_mask_path = '/content/brain-tumor-image-dataset-semantic-segmentation/train_masks'
    val_img_path = '/content/brain-tumor-image-dataset-semantic-segmentation/valid'
    val_mask_path = '/content/brain-tumor-image-dataset-semantic-segmentation/valid_masks'

    # --- Data Loading & Transforms ---
    image_transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])

    train_dataset = SemanticSegmentationDataset(train_img_path, train_mask_path, image_transforms=image_transforms,
                                                mask_transforms=mask_transforms)
    val_dataset = SemanticSegmentationDataset(val_img_path, val_mask_path, image_transforms=image_transforms,
                                              mask_transforms=mask_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, Optimizer, Scheduler ---
    model = Unet(
        backbone=backbone,
        in_channels=3,
        num_classes=1,
        pretrained=True,
        use_transformer_bottleneck=False  # Assuming transformer backbones
    )

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- Instantiate the correct loss function ---
    if loss_name == 'DiceLoss':
        criterion = DiceLoss(from_logits=True)
    elif loss_name == 'DiceBCELoss':
        criterion = DiceBCELoss()
    elif loss_name == 'FocalLoss':
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()),
        lr_scheduler=scheduler,
        epochs=15,  # You might want to use fewer epochs for a grid search
        save_dir=save_dir,
        resume_from=None  # Start each run from scratch
    )

    trainer.fit(train_loader, val_loader)

    # --- Return results for logging ---
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    best_loss = trainer.best_val_loss
    best_dice = 1.0 - best_loss

    return best_loss, best_dice, best_model_path


def grid_search():
    # --- Mount Google Drive ---
    # print("Mounting Google Drive...")
    # try:
    #     drive.mount('/content/gdrive', force_remount=True)
    #     print("Google Drive mounted successfully.")
    # except Exception as e:
    #     print(f"Error mounting Google Drive: {e}")
    #     return

    # --- Define main save directory in Google Drive ---
    gdrive_save_path = '/content/gdrive/My Drive/segmentation_model/segmentation_model'
    os.makedirs(gdrive_save_path, exist_ok=True)

    # --- DEFINE YOUR HYPERPARAMETER GRID HERE ---
    param_grid = {
        'backbone': [
            'resnet50',
            'swin_tiny_patch4_window7_224',
            'vit_base_patch16_224',
            'maxvit_tiny_tf_224.in1k',
            'beit_base_patch16_224'
        ],
        'img_size': [224],  # All these models have 224x224 pretrained weights
        'batch_size': [8],
        'lr': [1e-4],
        'loss_function': ['DiceLoss', 'DiceBCELoss', 'FocalLoss']
    }

    # --- Setup Results CSV in Google Drive ---
    results_file = os.path.join(gdrive_save_path, 'grid_search_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(param_grid.keys()) + ['best_val_loss', 'best_val_dice', 'best_model_path'])

    # --- Generate all combinations of parameters ---
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting Grid Search. Total runs: {len(param_combinations)}")

    # --- Loop through each combination and run training ---
    for params in param_combinations:
        is_transformer = any(x in params['backbone'] for x in ['swin', 'vit', 'beit', 'maxvit'])
        if is_transformer and params['img_size'] != 224:
            print(f"Skipping invalid combination for transformer: {params}")
            continue

        best_loss, best_dice, model_path = run_training(params, base_save_path=os.path.join(gdrive_save_path,
                                                                                            'grid_search_runs'))

        # Log the results
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(params.values()) + [best_loss, best_dice, model_path])

    print("\n--- Grid Search Finished! ---")
    print(f"Results have been saved to {results_file}")


if __name__ == '__main__':
    grid_search()
