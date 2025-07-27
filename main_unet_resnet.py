import os
import sys
import torch
from PIL import Image
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from models.custom_unet import Unet
# from models.losses import DiceLoss
# from utils.dataset import SemanticSegmentationDataset
# from utils.reproducibility import set_seed
# from utils.trainer import Trainer
from custom_unet import Unet
from losses import DiceLoss
from dataset import SemanticSegmentationDataset
from reproducibility import set_seed
from trainer import Trainer
from torch.utils.data import DataLoader, Dataset

def main():
    # --- Configuration ---
    set_seed(42)
    IMG_SIZE = 224
    BATCH_SIZE = 2  # As per your request
    EPOCHS = 15
    LR = 1e-4

    # --- Paths (replace with your actual dataset paths) ---
    # Using placeholder paths for demonstration
    train_img_path = '/content/brain-tumor-image-dataset-semantic-segmentation/train'
    train_mask_path = '/content/brain-tumor-image-dataset-semantic-segmentation/train_masks'
    val_img_path = '/content/brain-tumor-image-dataset-semantic-segmentation/valid'
    val_mask_path = '/content/brain-tumor-image-dataset-semantic-segmentation/valid_masks'

    # Create dummy data for testing if paths don't exist
    if not os.path.exists(train_img_path):
        print("Creating dummy dataset for demonstration...")
        os.makedirs(train_img_path, exist_ok=True);
        os.makedirs(train_mask_path, exist_ok=True)
        os.makedirs(val_img_path, exist_ok=True);
        os.makedirs(val_mask_path, exist_ok=True)
        for i in range(10):
            Image.new('RGB', (IMG_SIZE, IMG_SIZE)).save(os.path.join(train_img_path, f'img_{i}.jpg'))
            Image.new('L', (IMG_SIZE, IMG_SIZE), color=255 if i % 2 == 0 else 0).save(
                os.path.join(train_mask_path, f'mask_{i}.png'))
            Image.new('RGB', (IMG_SIZE, IMG_SIZE)).save(os.path.join(val_img_path, f'img_{i}.jpg'))
            Image.new('L', (IMG_SIZE, IMG_SIZE), color=255 if i % 2 != 0 else 0).save(
                os.path.join(val_mask_path, f'mask_{i}.png'))

    # --- Data Loading & Transforms ---
    image_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])

    train_dataset = SemanticSegmentationDataset(train_img_path, train_mask_path, image_transforms=image_transforms,
                                                mask_transforms=mask_transforms)
    val_dataset = SemanticSegmentationDataset(val_img_path, val_mask_path, image_transforms=image_transforms,
                                              mask_transforms=mask_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model, Optimizer, Scheduler ---
    model = Unet(
        backbone='swin_base_patch4_window7_224',  # Use Swin Transformer
        in_channels=3,
        num_classes=1,
        pretrained=True,
        use_transformer_bottleneck=False  # The backbone IS the transformer
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        criterion=DiceLoss(from_logits=True),  # Use from_logits=True as model outputs raw scores
        optimizer=optimizer,
        scaler=torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()),
        lr_scheduler=scheduler,
        epochs=EPOCHS,
        save_dir="./content/gdrive/My Drive/segmentation_model",
        resume_from="/content/gdrive/My Drive/segmentation_models/checkpoints/latest_model.pth"
    )

    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()