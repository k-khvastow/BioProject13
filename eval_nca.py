import torch
import torch.utils.data as data
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2

from src.models.NCA import FastSegNCA
from src.losses.LossFunctions import DiceLoss
from src.agents.Agent import Agent
from src.datasets.VideoBatchDataReader import Video3DDataset

DATA_ROOT = 'data/OCTA_6mm/OCT'
LABEL_ROOT = 'data/OCTA_6mm/GT_Layers'
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    'channel_n': 32,
    'hidden_size': 32,
    'steps': 16,
    'batch_size': 1,
    'epochs': 3,
    'resize': 128,
    'num_samples': 20,
}

class RobustVideoDataset(data.Dataset):
    def __init__(self, video_dataset, indices, resize=128):
        self.video_dataset = video_dataset
        self.resize = resize
        self.valid_indices = []
        
        print(f"Validating {len(indices)} samples...")
        for idx in indices:
            try:
                volume, mask = video_dataset[idx]
                min_frames = min(volume.shape[1], mask.shape[0])
                if min_frames > 0:
                    self.valid_indices.append(idx)
            except:
                pass
        print(f"{len(self.valid_indices)} valid samples")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        volume, mask = self.video_dataset[real_idx]
        
        min_frames = min(volume.shape[1], mask.shape[0])
        mid_frame = min_frames // 2
        
        img = volume[0, mid_frame, :, :].numpy()
        target = mask[mid_frame, :, :].numpy()
        
        img = cv2.resize(img, (self.resize, self.resize))
        target = cv2.resize(target.astype(np.float32), (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)
        
        img = img / 255.0
        img_3ch = np.stack([img, img, img], axis=-1)
        
        return torch.from_numpy(img_3ch).float(), torch.from_numpy((target > 0).astype(np.float32))

def dice_coefficient(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return (2. * intersection / union).item() if union > 0 else 1.0

def iou_score(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / union).item() if union > 0 else 1.0

class QuickAgent(Agent):
    def prepare_data(self, data, eval=False):
        inputs, targets = data
        inputs = self.make_seed(inputs)
        inputs = inputs.permute(0, 3, 1, 2)
        return inputs, targets.to(self.device)

def main():
    print("Evaluation\n")
    torch.cuda.empty_cache()
    
    full_dataset = Video3DDataset(DATA_ROOT, LABEL_ROOT, preload=False)
    indices = list(range(min(CONFIG['num_samples'], len(full_dataset))))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    train_dataset = RobustVideoDataset(full_dataset, train_idx, CONFIG['resize'])
    test_dataset = RobustVideoDataset(full_dataset, test_idx, CONFIG['resize'])
    
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}\n")
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSegNCA(channel_n=32, hidden_size=32, input_channels=3).to(device)
    agent = QuickAgent(model, steps=16, channel_n=32, batch_size=1)
    loss_fn = DiceLoss()
    
    print("Training...")
    for epoch in range(CONFIG['epochs']):
        losses = []
        for batch in train_loader:
            torch.cuda.empty_cache()
            loss = agent.batch_step(batch, lambda: loss_fn, train=True)
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Loss={np.mean(losses):.4f}")
    
    torch.save(model.state_dict(), OUTPUT_DIR / 'model.pth')
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    dice_scores, iou_scores, times = [], [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            torch.cuda.empty_cache()
            seed = torch.zeros(1, CONFIG['resize'], CONFIG['resize'], 32).to(device)
            seed[..., :3] = inputs.to(device)
            seed = seed.permute(0, 3, 1, 2)
            
            torch.cuda.synchronize()
            start = time.time()
            output, _ = model(seed, steps=16)
            torch.cuda.synchronize()
            times.append(time.time() - start)
            
            dice_scores.append(dice_coefficient(output, targets.to(device)))
            iou_scores.append(iou_score(output, targets.to(device)))
    
    # Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Dice:      {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU:       {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Inference: {np.mean(times)*1000:.2f} ms")
    print(f"FPS:       {1/np.mean(times):.2f}")
    print("="*50)
    
    # Save
    with open(OUTPUT_DIR / 'report.txt', 'w') as f:
        f.write("NCA Video Segmentation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Config: {CONFIG['channel_n']}ch, {CONFIG['steps']}steps, {CONFIG['resize']}px\n")
        f.write(f"Training: {CONFIG['epochs']} epochs, {len(train_dataset)} samples\n\n")
        f.write(f"Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}\n")
        f.write(f"IoU Score:  {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}\n")
        f.write(f"Inference:  {np.mean(times)*1000:.2f} ms\n")
        f.write(f"FPS:        {1/np.mean(times):.2f}\n")

    print("\nDone, report saved to /results/report.txt")

if __name__ == "__main__":
    main()