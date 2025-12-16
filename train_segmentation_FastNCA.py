import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

# Project specific imports
from src.models.NCA import FastSegNCA
from src.losses.LossFunctions import DiceLoss
from src.agents.Agent import Agent
from src.utils.utils import compute_dataset_mean_std

@dataclass
class Config:
    image_dir: Path = Path("/vol/data/BioProject13/acevedo/images")
    mask_dir: Path = Path("/vol/data/BioProject13/acevedo/masks")
    output_dir: Path = Path("/vol/data/BioProject13/output_acevedo")
    
    target_type: str = "nuclei"  # 'cell' or 'nuclei'
    resize: int = 64
    batch_size: int = 8
    channel_n: int = 128
    hidden_size: int = 128
    steps: int = 64
    learning_rate: float = 0.0004
    epochs: int = 32

def find_paired_paths(config: Config) -> Tuple[List[Path], List[Path]]:
    valid_images = []
    valid_masks = []
    supported_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

    for img_path in config.image_dir.rglob('*'):
        if img_path.suffix.lower() not in supported_exts:
            continue

        rel_path = img_path.relative_to(config.image_dir).parent
        mask_filename = f"{img_path.stem}-{config.target_type}-0.jpg"
        mask_path = config.mask_dir / rel_path / mask_filename

        if mask_path.exists():
            valid_images.append(img_path)
            valid_masks.append(mask_path)

    return valid_images, valid_masks

class SegmentationDataset(data.Dataset):
    def __init__(self, images: List[Path], masks: List[Path], resize: int, mean: List[float], std: List[float]):
        self.images = images
        self.masks = masks
        self.resize = resize
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            img = img.convert("RGB").resize((self.resize, self.resize))
            mask = mask.convert("L").resize((self.resize, self.resize), resample=Image.NEAREST)

            img_np = np.array(img)
            mask_np = np.array(mask)

        img_tensor = self.transform(img_np)
        
        # Binarize and normalize mask
        mask_tensor = torch.from_numpy(np.round(mask_np / 255.0).astype(np.float32))

        # Permute for NCA format (H, W, C)
        return img_tensor.permute(1, 2, 0), mask_tensor

class TrainingAgent(Agent):
    def train(self, train_loader, val_loader, loss_func, epochs, name, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_history = []
        val_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            total_batches = len(train_loader)

            # Training Loop
            for i, batch_data in enumerate(train_loader):
                epoch_loss += self.batch_step(batch_data, loss_func)
                progress = ((i + 1) / total_batches) * 100
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {i+1}/{total_batches} | {progress:.1f}%", end="")

            avg_train_loss = (epoch_loss / total_batches).item()
            train_history.append(avg_train_loss)

            # Validation Loop
            val_loss = 0.0
            for batch_data in val_loader:
                val_loss += self.batch_step(batch_data, loss_func, train=False)
            
            avg_val_loss = (val_loss / len(val_loader)).item()
            val_history.append(avg_val_loss)

            print(f"\rEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}      ")

        self._save_results(train_history, val_history, name, output_dir)

    def _save_results(self, train_loss, val_loss, name, output_dir):
        np.savetxt(output_dir / f"{name}_loss.csv", [train_loss, val_loss], delimiter=',')
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Val")
        plt.legend()
        plt.title(f'Loss Curve: {name}')
        plt.savefig(output_dir / f"{name}_loss_plot.png")
        plt.close()

    def prepare_data(self, data, eval=False):
        # 1. Let the parent Agent prepare the seed (returns B, H, W, C on GPU)
        inputs, targets = super().prepare_data(data, eval)
        
        # 2. Permute inputs to (Batch, Channel, Height, Width) for FastSegNCA
        inputs = inputs.permute(0, 3, 1, 2)
        
        return inputs, targets

def main():
    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data Preparation
    images, masks = find_paired_paths(cfg)
    if not images:
        raise FileNotFoundError(f"No pairs found for target '{cfg.target_type}' in {cfg.image_dir}")
    
    print(f"Found {len(images)} pairs.")

    mean, std = compute_dataset_mean_std(images, cfg.resize)
    
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    train_ds = SegmentationDataset(X_train, y_train, cfg.resize, mean, std)
    val_ds = SegmentationDataset(X_test, y_test, cfg.resize, mean, std)

    train_loader = data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False)

    # Model Initialization
    model = FastSegNCA(channel_n=cfg.channel_n, hidden_size=cfg.hidden_size, input_channels=3)
    model.to(device)

    agent = TrainingAgent(model, steps=cfg.steps, channel_n=cfg.channel_n, batch_size=cfg.batch_size)
    
    # Execution
    run_name = f"Acevedo_FastSeg_{cfg.target_type}"
    agent.train(train_loader, val_loader, DiceLoss, cfg.epochs, run_name, cfg.output_dir)
    
    save_path = cfg.output_dir / f"{run_name}_{cfg.epochs}epochs.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

if __name__ == "__main__":
    main()