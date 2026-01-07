import torch
import torch.utils.data as data
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import os

from src.models.NCA import FastSegNCA
from src.datasets.VideoBatchDataReader import Video3DDataset

# Configuration
DATA_ROOT = 'data/OCTA_6mm/OCT'
LABEL_ROOT = 'data/OCTA_6mm/GT_Layers'
OUTPUT_DIR = Path('results_viz')
OUTPUT_DIR.mkdir(exist_ok=True)

def create_overlap_visualization(img, gt_mask, pred_mask):
    """
    Creates a visualization with masks overlaid on the original image.
    Handles Multi-class GT (0-7) and Binary Prediction (0-1).
    
    Visualization Scheme:
    - Ground Truth Layers (1-7): Gradient of Cyan->Green->Yellow shades.
    - Prediction (1): Red transparent overlay.
    
    Interaction:
    - GT Only (FN): Cyan/Green pixels (The model missed these).
    - Pred Only (FP): Red pixels (The model hallucinated these).
    - Overlap (TP): Mix of GT color and Red (Orange/Brownish) - Agreement.
    """
    # img: (H, W) or (H, W, 3), range [0, 1] or [0, 255]
    # gt_mask: (H, W), int 0-7
    # pred_mask: (H, W), binary 0 or 1
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
        
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    height, width = img_color.shape[:2]
    
    # 1. Prepare GT Overlay (Multi-color)
    # Map classes 1-7 to colors.
    # We use a colormap logic:
    # 0: Black (Transparent)
    # 1-7: HSL spectrum or cool colors to contrast with Red prediction.
    # Let's use shades of Blue/Green/Cyan for GT to contrast with Red Pred.
    
    gt_overlay = np.zeros_like(img_color)
    
    # Simple static map for 7 layers
    # BGR format
    colors = [
        (0, 0, 0),       # 0: Bg
        (255, 0, 0),     # 1: Blue
        (255, 128, 0),   # 2: Azure
        (255, 255, 0),   # 3: Cyan
        (128, 255, 0),   # 4: Spring Green
        (0, 255, 0),     # 5: Green
        (0, 255, 128),   # 6: ...
        (0, 255, 255),   # 7: Yellow
        # If more layers, wrap or add more
    ]
    
    # This loop is slow for python but safe for 128x128. 
    # Faster: use indexing.
    for c in range(1, 8):
        # Mask for class c
        if c < len(colors):
            matches = (gt_mask == c)
            if matches.any():
                gt_overlay[matches] = colors[c]

    # 2. Prepare Pred Overlay (Red)
    pred_overlay = np.zeros_like(img_color)
    # Red: (0, 0, 255)
    pred_overlay[pred_mask == 1] = (0, 0, 255)
    
    # 3. Combine
    # We want:
    # Base Image
    # + GT (alpha=0.3)
    # + Pred (alpha=0.3)
    
    # But simply adding them might wash out.
    # Let's update the image where masks exist.
    
    final = img_color.astype(np.float32)
    
    # Add GT
    gt_indices = np.any(gt_overlay > 0, axis=-1)
    if gt_indices.any():
        # Blend: 0.6 * Base + 0.4 * GT
        final[gt_indices] = final[gt_indices] * 0.6 + gt_overlay[gt_indices] * 0.4
        
    # Add Pred
    pred_indices = (pred_mask == 1)
    if pred_indices.any():
        # Blend: Current + Red.
        # If we already added GT, this mixes Red into it.
        # Current * 0.6 + Red * 0.4
        final[pred_indices] = final[pred_indices] * 0.6 + pred_overlay[pred_indices] * 0.4
        
    return final.astype(np.uint8)

def run_visualization(model, volume, mask, device, config, output_path):
    frames = volume[0] # (Frames, H, W)
    num_frames = frames.shape[0]
    height, width = frames.shape[1], frames.shape[2]
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
    
    hidden_state = None
    params = {
        'steps': config['full_steps'],
        'warm_steps': config['warm_steps'],
        'resize': config['resize']
    }
    
    print(f"Generating video to {output_path}...")
    
    with torch.inference_mode():
        for t in range(num_frames):
            img_np = frames[t]
            target_np = mask[t] # (H, W) - INT CLASSES 0-7
            
            # Prepare Input
            # Resize if needed (for model, not visually if possible, but simplest to assume model input size)
            if params['resize'] is not None:
                h_in, w_in = params['resize'], params['resize']
                img_in = cv2.resize(img_np, (h_in, w_in))
            else:
                img_in = img_np
                
            img_norm = img_in / 255.0
            if len(img_norm.shape) == 2:
                img_3ch = np.stack([img_norm]*3, axis=-1)
            else:
                img_3ch = img_norm
            
            input_tensor = torch.from_numpy(img_3ch).float().to(device)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Warm Start Logic
            if t == 0 or hidden_state is None:
                current_steps = params['steps']
                h, w = input_tensor.shape[2], input_tensor.shape[3]
                seed = torch.zeros(1, config['channel_n'], h, w, device=device)
                seed[:, :3, :, :] = input_tensor
                current_state = seed
            else:
                current_steps = params['warm_steps']
                current_state = hidden_state.clone()
                current_state[:, :3, :, :] = input_tensor
                
            # Inference
            output_mask, final_state = model(current_state, steps=current_steps)
            hidden_state = final_state
            
            # Process Output
            pred_prob = torch.sigmoid(output_mask).cpu().numpy()[0, 0] # (H, W)
            pred_bin = (pred_prob > 0.5).astype(np.uint8)
            
            # Helper to resize INT masks (NN interpolation)
            def resize_mask(m, size):
                return cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_NEAREST).astype(np.int64)

            # Resize GT back to original if needed? 
            # Or assume we output at 'resize' resolution.
            # Current `img_in` is resized resolution. Let's use that.
            
            # target_np is original size.
            if params['resize'] is not None and (target_np.shape[0] != img_in.shape[0]):
                 gt_resized = resize_mask(target_np, (img_in.shape[1], img_in.shape[0]))
            else:
                 gt_resized = target_np.astype(np.int64)

            # Create Frame
            # gt_resized contains 0-7 classes.
            viz_frame = create_overlap_visualization(img_in, gt_resized, pred_bin)
            
            # Write
            out_video.write(viz_frame)
            
            if t % 10 == 0:
                print(f"Frame {t}/{num_frames}", end='\r')
                
    out_video.release()
    print(f"\nSaved {output_path}")

def main():
    torch.set_float32_matmul_precision('high')
    
    # Load Dataset
    print("Loading Dataset...")
    full_dataset = Video3DDataset(DATA_ROOT, LABEL_ROOT, preload=False)
    
    # Pick a sample
    # Let's pick sample index 0 from the list (or random)
    sample_idx = 0 
    if len(full_dataset) == 0:
        print("No data found.")
        return
        
    print(f"Visualizing Sample Index: {sample_idx} ID: {full_dataset.sample_ids[sample_idx]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model_path = Path('results') / 'model.pth'
    if not model_path.exists():
        print("Model not found.")
        return
        
    model = FastSegNCA(channel_n=CONFIG['channel_n'], 
                       hidden_size=CONFIG['hidden_size'], 
                       input_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get Data
    volume, mask = full_dataset[sample_idx]
    vol_np = volume.numpy()
    mask_np = mask.numpy()
    
    # Run
    output_filename = OUTPUT_DIR / f"viz_{full_dataset.sample_ids[sample_idx]}.mp4"
    run_visualization(model, vol_np, mask_np, device, CONFIG, output_filename)

if __name__ == "__main__":
    main()
