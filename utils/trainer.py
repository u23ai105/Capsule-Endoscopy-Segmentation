import os
import csv
import torch
from torch import autocast
from tqdm import tqdm


# ====================================================================================
# Trainer Class (MODIFIED to include CSV logging)
# ====================================================================================
class Trainer:
    def __init__(self, model, criterion, optimizer, scaler, lr_scheduler, epochs, save_dir, resume_from=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.save_dir = save_dir
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        os.makedirs(self.save_dir, exist_ok=True)

        # **NEW**: Initialize log file
        self.log_file = os.path.join(self.save_dir, 'training_log.csv')
        self._init_log_file()

        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)

    def _init_log_file(self):
        # **NEW**: Creates the log file and writes the header if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice_score', 'learning_rate'])

    def _log_epoch(self, epoch, train_loss, val_loss, val_dice):
        # **NEW**: Appends a new row of metrics to the log file
        current_lr = self.optimizer.param_groups[0]['lr']
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_dice, current_lr])

    def _load_checkpoint(self, path):
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.lr_scheduler and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {self.start_epoch}. Best validation loss: {self.best_val_loss:.4f}")

    def _save_checkpoint(self, epoch, is_best=False):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(), 'best_val_loss': self.best_val_loss}
        if self.lr_scheduler: state['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        latest_path = os.path.join(self.save_dir, 'latest_model.pth')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(state, best_path)
            print(f"Saved new best model to {best_path}")

    def _run_epoch(self, data_loader, is_training=True):
        self.model.train() if is_training else self.model.eval()
        total_loss = 0.0
        total_dice = 0.0  # **NEW**: For validation dice score

        desc = "Training" if is_training else "Validation"
        progress_bar = tqdm(data_loader, desc=desc, leave=False)

        for images, masks in progress_bar:
            images, masks = images.to(self.device), masks.to(self.device)
            with torch.set_grad_enabled(is_training):
                with autocast(device_type=self.device, enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                if is_training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # **NEW**: Calculate dice score for validation
                    dice_score = 1.0 - loss.item()
                    total_dice += dice_score

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(data_loader)
        avg_dice = total_dice / len(data_loader) if not is_training else 0
        return avg_loss, avg_dice

    def fit(self, train_loader, val_loader):
        print("Starting training...")
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            train_loss, _ = self._run_epoch(train_loader, is_training=True)
            print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            val_loss, val_dice = self._run_epoch(val_loader, is_training=False)
            print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f} | Validation Dice Score: {val_dice:.4f}")

            if self.lr_scheduler: self.lr_scheduler.step(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch, is_best=is_best)
            self._log_epoch(epoch, train_loss, val_loss, val_dice)  # **NEW**

        print("\nTraining finished!")
