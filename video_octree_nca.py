import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# From your existing BioProject13 codebase
from src.datasets.VideoBatchDataReader import Video3DDataset
from src.models.NCA import FastSegNCA

# --- Minimal Octree Implementation (extracted from OctreeNCA) ---
class Octree:
    @torch.no_grad()
    def __init__(self, init_batch: torch.Tensor, input_channels: int) -> None:
        self.input_channels = input_channels

        assert init_batch.ndim == 4, "init_batch must be BHWC tensor"
        assert init_batch.shape[1] == init_batch.shape[2], "init_batch must be square"
        
        self.levels_of_detail = [init_batch]
        while self.levels_of_detail[-1].shape[1] > 16:
            # create temp with BCHW order
            temp = self.levels_of_detail[-1].permute(0, 3, 1, 2)
            lower_res = F.avg_pool2d(temp, 2)
            lower_res = lower_res.permute(0, 2, 3, 1)
            self.levels_of_detail.append(lower_res)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, depth: int):
        """Create octree from a 4D tensor (BHWC or BTHW format)"""
        # For now, treat as BHWC and create octree structure
        if tensor.ndim == 5:  # (B, T, H, W, C) - video format
            # Reshape to (B*T, H, W, 1)
            b, t, h, w = tensor.shape[:4]
            tensor = tensor.reshape(b*t, h, w, 1)
        
        octree = cls(tensor, input_channels=tensor.shape[-1] if tensor.ndim == 4 else 1)
        # Initialize features from the first input channel
        octree.features = tensor[..., :1].reshape(-1, 1)  # Flatten spatial dims
        return octree

    def plot(self, output_path: str = 'octree.pdf') -> None:
        fig, axs = plt.subplots(1, len(self.levels_of_detail), figsize=(20, 20))
        for i, img in enumerate(self.levels_of_detail):
            axs[i].imshow(img[0, :, :, 0].cpu(), cmap='gray')
        plt.savefig(output_path, bbox_inches='tight')

    def upscale_states(self, from_level: int) -> None:
        assert from_level in range(1, len(self.levels_of_detail)), "from_level must be in range(1, len(levels_of_detail))"
        temp = self.levels_of_detail[from_level]
        temp = temp[..., self.input_channels:]
        temp = temp.permute(0, 3, 1, 2) # BHWC -> BCHW
        upsampled_states = torch.nn.Upsample(scale_factor=2, mode='nearest')(temp)
        upsampled_states = upsampled_states.permute(0, 2, 3, 1) # BCHW -> BHWC

        self.levels_of_detail[from_level-1] = torch.cat([self.levels_of_detail[from_level-1][..., :self.input_channels], upsampled_states], dim=-1)

    def to(self, device):
        """Move octree to device"""
        self.levels_of_detail = [level.to(device) for level in self.levels_of_detail]
        if hasattr(self, 'features'):
            self.features = self.features.to(device)
        return self

    @property
    def n_nodes(self):
        """Get total number of nodes (approximate from levels)"""
        # Sum of all nodes across octree levels
        n = 0
        for level in self.levels_of_detail:
            n += level.shape[0] * level.shape[1] * level.shape[2]
        return n

    def to_tensor(self):
        """Convert octree back to dense tensor"""
        return self.levels_of_detail[0]

# --- Model Wrapper for Octree + FastSegNCA ---
class OctreeFastSegNCA(torch.nn.Module):
    """Wrapper that applies FastSegNCA within an octree sparse structure"""
    def __init__(self, channel_n, hidden_size, device="cpu"):
        super().__init__()
        self.device_str = str(device)
        # Use the real FastSegNCA architecture
        self.nca = FastSegNCA(
            channel_n=channel_n,
            fire_rate=0.5,
            device=self.device_str,
            hidden_size=hidden_size,
            input_channels=channel_n,  # Match the number of channels to channel_n
            init_method="standard"
        )
        # Ensure model is on the correct device
        self.target_device = torch.device(device)
        self.nca = self.nca.to(self.target_device)

    def forward(self, x, steps=16, fire_rate=0.5):
        """
        Apply FastSegNCA to the input tensor
        x: (B, C, H, W) - expected to be in BCHW format for FastSegNCA
        """
        # Ensure input is on the correct device
        x = x.to(self.target_device)
        out, state = self.nca(x, steps=steps, fire_rate=fire_rate)
        return out, state

# --- Configuration ---
DATA_ROOT = 'data/OCTA_6mm/OCT'
LABEL_ROOT = 'data/OCTA_6mm/GT_Layers'
OUTPUT_DIR = Path('results_octree')
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    'channel_n': 32,
    'hidden_size': 32,
    'full_steps': 4,      # Reduced from 16 to save memory
    'octree_depth': 7,    # Depth 7 = 128x128x128. Change to 8 for 256.
    'num_samples': 20,     # Number of videos to test
}

# Placeholder NCALayer - unused since OctreeFastSegNCA is used
class NCALayer(torch.nn.Module):
    def __init__(self, channel_n: int, hidden_size: int):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_size = hidden_size
        # TODO: Implement NCALayer properly from OctreeNCA sources
        
    def forward(self, features, octree):
        # TODO: Implement forward pass
        return features

# --- Configuration ---
def pad_to_power_of_2(tensor):
    c, t, h, w = tensor.shape
    max_dim = max(t, h, w)
    new_size = 2**(max_dim - 1).bit_length()
    
    pad_t, pad_h, pad_w = new_size - t, new_size - h, new_size - w
    padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_t))
    return padded, (t, h, w)

def dice_coefficient(pred, target, threshold=0.5):
    # Pred is likely logits, apply sigmoid
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return (2. * intersection / union).item() if union > 0 else 1.0

def iou_score(pred, target, threshold=0.5):
    # Intersection over Union
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / union).item() if union > 0 else 1.0

# --- Evaluation Logic ---
def evaluate_video_octree(model, volume, mask, device):
    # volume: (1, T, H, W) - typically (1, num_frames, height, width)
    # mask: (T, H, W)
    video_tensor = volume.to(device)
    
    # 1. Downsampling to reduce memory usage
    # Reduce by factor of 2 to fit in GPU memory
    if video_tensor.shape[2] > 256:
        video_tensor = F.interpolate(
            video_tensor.unsqueeze(0),  # Add batch dim: (1, 1, T, H, W)
            scale_factor=0.5,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dim
    
    t, h, w = video_tensor.shape[1:]
    frames_tensor = video_tensor[0]  # (T, H, W)
    
    start_time = time.time()
    
    # 3. Model Inference - process frames one at a time to avoid memory issues
    output_frames = []
    with torch.inference_mode():
        # Initialize state for temporal coherence (warm start)
        state = None
        for frame_idx in range(t):
            frame = frames_tensor[frame_idx]  # (H, W)
            
            # Pad single channel to match channel_n
            frame_bchw = frame.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            frame_bchw = frame_bchw.expand(1, CONFIG['channel_n'], -1, -1)  # (1, 32, H, W)
            
            # Apply model (cold start for all frames in octree version)
            out, state = model(frame_bchw, steps=CONFIG['full_steps'], fire_rate=0.5)
            output_frames.append(out[0])  # Extract (H, W)
    
    # Concatenate all output frames: (T, H, W)
    output_volume = torch.stack(output_frames, dim=0)
    
    # Upsample mask to match output size if it was downsampled
    if mask.shape != output_volume.shape:
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),  # (1, 1, T, H, W)
            size=output_volume.shape,
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # Back to (T, H, W)
    
    total_time = time.time() - start_time
    dice = dice_coefficient(output_volume, mask.to(device))
    iou = iou_score(output_volume, mask.to(device))
    
    return dice, iou, total_time

# --- Main Execution ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running OctreeNCA Evaluation on {device}")
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Load Dataset
    dataset = Video3DDataset(DATA_ROOT, LABEL_ROOT, preload=False)
    
    # 2. Initialize Model
    # Note: In a real scenario, you'd load weights here: model.load_state_dict(...)
    model = OctreeFastSegNCA(CONFIG['channel_n'], CONFIG['hidden_size'], device=str(device)).to(device)
    model.eval()

    all_dice = []
    all_iou = []
    all_times = []

    print(f"{'Video ID':<10} | {'Dice Score':<12} | {'IoU Score':<12} | {'Time (s)':<10}")
    print("-" * 55)

    for i in range(min(CONFIG['num_samples'], len(dataset))):
        try:
            # volume: (1, T, H, W), mask: (T, H, W)
            volume, mask = dataset[i]
            
            dice, iou, duration = evaluate_video_octree(model, volume, mask, device)
            all_dice.append(dice)
            all_iou.append(iou)
            all_times.append(duration)
            
            print(f"{i:<10} | {dice:<12.4f} | {iou:<12.4f} | {duration:<10.2f}")
            
        except Exception as e:
            print(f"Error on video {i}: {e}")

    print("-" * 55)
    
    # Calculate and display final results
    print("\n" + "="*55)
    print("FINAL RESULTS - OCTREE NCA MODE")
    print("="*55)
    if all_dice:
        mean_dice = np.mean(all_dice)
        mean_iou = np.mean(all_iou)
        mean_time = np.mean(all_times)
        std_dice = np.std(all_dice)
        std_iou = np.std(all_iou)
        
        print(f"Mean Dice:      {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Mean IoU:       {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"Mean Inference: {mean_time*1000:.2f} ms")
        print(f"Mean FPS:       {1/mean_time:.2f}")
        
        # Save Report
        with open(OUTPUT_DIR / 'octree_report.txt', 'w') as f:
            f.write("OctreeNCA Video Segmentation Results\n")
            f.write("="*55 + "\n")
            f.write(f"Config: {CONFIG}\n")
            f.write("="*55 + "\n")
            f.write(f"Mean Dice:      {mean_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Mean IoU:       {mean_iou:.4f} ± {std_iou:.4f}\n")
            f.write(f"Mean Inference: {mean_time*1000:.2f} ms\n")
            f.write(f"Mean FPS:       {1/mean_time:.2f}\n")
    else:
        print("No videos processed successfully.")
    print("="*55)

if __name__ == "__main__":
    main()