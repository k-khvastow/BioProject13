import torch
import torch.utils.data as data
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2

from src.models.NCA import FastSegNCA
from src.datasets.VideoBatchDataReader import Video3DDataset

DATA_ROOT = 'data/OCTA_6mm/OCT'
LABEL_ROOT = 'data/OCTA_6mm/GT_Layers'
OUTPUT_DIR = Path('results_video')
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    'channel_n': 32,
    'hidden_size': 32,
    'full_steps': 16,     # Steps for the first frame (cold start)
    'warm_steps': 4,      # Steps for subsequent frames (warm start)
    'batch_size': 1,
    'resize': None,       # Set to None to use original resolution, or an integer (e.g. 128)
    'num_samples': 20,    # Total videos to evaluate
}

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

def evaluate_video(model, volume, mask, device, config):
    """
    Evaluates the model on a single video sequence using temporal coherence.
    """
    # volume: (1, Frames, H, W) -> numpy
    # mask: (Frames, H, W) -> numpy
    
    frames = volume[0] # (Frames, H, W)
    num_frames = frames.shape[0]
    
    # Pre-allocate metrics
    video_dice = []
    video_iou = []
    video_times = []
    
    hidden_state = None
    
    params = {
        'steps': config['full_steps'],
        'warm_steps': config['warm_steps'],
        'resize': config.get('resize')
    }
    
    # Use torch.inference_mode for slightly faster inference than no_grad
    with torch.inference_mode():
        for t in range(num_frames):
            # 1. Prepare Input Frame
            img_np = frames[t] # (H, W) or (H, W, C)
            target_np = mask[t]
            
            # Resize if requested
            if params['resize'] is not None:
                resize_dim = (params['resize'], params['resize'])
                img_resized = cv2.resize(img_np, resize_dim)
                target_resized = cv2.resize(target_np.astype(np.float32), resize_dim, interpolation=cv2.INTER_NEAREST)
            else:
                img_resized = img_np
                target_resized = target_np.astype(np.float32)
            
            # Normalization & Shape
            img_norm = img_resized / 255.0
            # Convert to 3 channels if grayscale
            if len(img_norm.shape) == 2:
                img_3ch = np.stack([img_norm]*3, axis=-1)
            else:
                img_3ch = img_norm
            
            # To Tensor: (1, 3, H, W)
            input_tensor = torch.from_numpy(img_3ch).float().to(device) # (H, W, 3)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
            
            target_tensor = torch.from_numpy((target_resized > 0).astype(np.float32)).to(device)
            
            # 2. State Initialization / Update
            if t == 0 or hidden_state is None:
                # Cold Start
                current_steps = params['steps']
                
                # Initialize seed state: (1, C, H, W)
                # Channel 0-2: RGB image
                # Channel 3+: Hidden zero
                # Use current frame dimensions
                h, w = input_tensor.shape[2], input_tensor.shape[3]
                seed = torch.zeros(1, config['channel_n'], h, w, device=device)
                seed[:, :3, :, :] = input_tensor
                
                current_state = seed
            else:
                # Warm Start
                current_steps = params['warm_steps']
                
                # Take previous hidden state
                current_state = hidden_state.clone()
                # ENFORCE HARD CONSTRAINT: Overwrite visible channels with NEW frame
                current_state[:, :3, :, :] = input_tensor
            
            # 3. Model Forward
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            # Run NCA
            output_mask, final_state = model(current_state, steps=current_steps)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            step_time = time.time() - start_time
            video_times.append(step_time)
            
            # 4. Save State for Next Frame
            hidden_state = final_state
            
            # 5. Metrics
            video_dice.append(dice_coefficient(output_mask, target_tensor))
            video_iou.append(iou_score(output_mask, target_tensor))
            
    return np.mean(video_dice), np.mean(video_iou), np.mean(video_times)

def main():
    print("Sequential Video NCA Evaluation (Warm Start)\n")
    torch.cuda.empty_cache()
    
    # 1. Setup Data
    full_dataset = Video3DDataset(DATA_ROOT, LABEL_ROOT, preload=False)
    indices = list(range(min(CONFIG['num_samples'], len(full_dataset))))
    
    # Use same split logic to respect train/test, though we only eval here
    _, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    print(f"Total Test Videos: {len(test_idx)}")
    print(f"Config: {CONFIG}\n")

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load weights from previous training
    model_path = Path('results') / 'model.pth'
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}. Please run eval_nca.py first to train the model.")
        return

    model = FastSegNCA(channel_n=CONFIG['channel_n'], 
                       hidden_size=CONFIG['hidden_size'], 
                       input_channels=3).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Evaluation Loop
    all_dice = []
    all_iou = []
    all_times = []
    
    print("Starting evaluation...")
    for i, idx in enumerate(test_idx):
        try:
            # Load full video volume
            volume, mask = full_dataset[idx] 
            # volume is Tensor (1, Frames, D, W) or numpy? check dataset
            # Video3DDataset returns (Tensor(1, F, D, W), Tensor(F, D, W))
            
            # Convert to numpy for easier resizing in the loop
            vol_np = volume.numpy() # (1, Frames, D, W)
            mask_np = mask.numpy()
            
            dice, iou, avg_time = evaluate_video(model, vol_np, mask_np, device, CONFIG)
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_times.append(avg_time)
            
            print(f"Video {idx}: Dice={dice:.4f}, IoU={iou:.4f}, AvgInference={avg_time*1000:.1f}ms (FPS: {1/avg_time:.1f})")
            
        except Exception as e:
            print(f"Error processing video {idx}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Final Aggregation
    print("\n" + "="*50)
    print("FINAL RESULTS - SEQUENTIAL MODE")
    print("="*50)
    if all_dice:
        mean_dice = np.mean(all_dice)
        mean_iou = np.mean(all_iou)
        mean_time = np.mean(all_times)
        
        print(f"Mean Dice:      {mean_dice:.4f} ± {np.std(all_dice):.4f}")
        print(f"Mean IoU:       {mean_iou:.4f} ± {np.std(all_iou):.4f}")
        print(f"Mean Inference: {mean_time*1000:.2f} ms")
        print(f"Mean FPS:       {1/mean_time:.2f}")
    else:
        print("No videos processed successfully.")
    print("="*50)
    
    # Save Report
    with open(OUTPUT_DIR / 'video_report.txt', 'w') as f:
        f.write("NCA Sequential Video Segmentation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Config: {CONFIG}\n")
        f.write(f"Mean Dice:      {mean_dice:.4f}\n")
        f.write(f"Mean IoU:       {mean_iou:.4f}\n")
        f.write(f"Mean FPS:       {1/mean_time:.2f}\n")

if __name__ == "__main__":
    main()
