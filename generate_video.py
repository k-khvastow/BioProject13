import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.datasets.VideoBatchDataReader import Video3DDataset

def generate_video():
    data_root = './data/OCTA_6mm/OCT'
    label_root = './data/Label/GT_Layers'
    output_file = './data/segmentation_video.mp4'
    
    print("Initializing dataset...")
    dataset = Video3DDataset(data_root, label_root, preload=False)
    
    print("Loading data (Sample 0)...")
    # data: (1, Frames, Depth, Width) -> (1, 400, 640, 400)
    # label: (Frames, Depth, Width) -> (400, 640, 400)
    data_tensor, label_tensor = dataset[0]
    
    # Convert to numpy
    # Remove batch dim from data
    data = data_tensor[0].numpy() # (400, 640, 400)
    label = label_tensor.numpy() # (400, 640, 400)
    
    frames, height, width = data.shape
    print(f"Volume shape: {frames}x{height}x{width}")
    
    # Initialize VideoWriter
    # Codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    # VideoWriter expects (width, height)
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Create a colormap
    # We have 7 classes (0-6). Let's use 'jet' or similar.
    cmap = plt.get_cmap('jet')
    
    print("Generating video frames...")
    
    for i in range(frames):
        if i % 50 == 0:
            print(f"Processing frame {i}/{frames}")
            
        # Get image and label for this frame
        img = data[i, :, :] # (640, 400)
        lbl = label[i, :, :] # (640, 400)
        
        # Normalize image to 0-255 uint8
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        
        # Normalize label to 0-1 for colormap
        # We have classes 0-6.
        lbl_norm = lbl / 6.0
        
        # Apply colormap
        # cmap returns RGBA, we want BGR 0-255
        heatmap = cmap(lbl_norm) # (640, 400, 4)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        # Blend
        alpha = 0.3
        blended = cv2.addWeighted(img_color, 1 - alpha, heatmap, alpha, 0)
        
        # Write frame
        out.write(blended)
        
    out.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    generate_video()
