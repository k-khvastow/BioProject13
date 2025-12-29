import os
import natsort
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import cv2
import scipy.io

class Video3DDataset(Dataset):
    def __init__(self, data_root, label_root, preload=False):
        """
        Args:
            data_root (str): Root directory containing subdirectories of image sequences (e.g., .../OCT).
            label_root (str): Root directory containing .mat files (e.g., .../GT_Layers).
            preload (bool): If True, load all data into RAM (Warning: High memory usage).
        """
        self.data_root = data_root
        self.label_root = label_root
        self.preload = preload
        
        # 1. Scan for valid data pairs
        self.sample_ids = []
        
        # List all subdirectories in data_root
        if not os.path.exists(data_root):
             raise ValueError(f"Data root does not exist: {data_root}")
             
        candidates = natsort.natsorted(os.listdir(data_root))
        
        for item in candidates:
            data_path = os.path.join(data_root, item)
            if os.path.isdir(data_path):
                # Check if corresponding label file exists
                label_path = os.path.join(label_root, f"{item}.mat")
                if os.path.exists(label_path):
                    self.sample_ids.append(item)
        
        print(f"Found {len(self.sample_ids)} valid samples.")
        
        if len(self.sample_ids) == 0:
            print(f"Warning: No valid samples found in {data_root} with labels in {label_root}")

        # 2. Preload if requested
        self.cache = {}
        if self.preload:
            print("Preloading data... (This may take a while and consume a lot of RAM)")
            for i in range(len(self.sample_ids)):
                self.cache[i] = self.load_sample(i)
                if i % 10 == 0:
                    print(f"Loaded {i}/{len(self.sample_ids)}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        if self.preload:
            return self.cache[index]
        else:
            return self.load_sample(index)

    def load_sample(self, index):
        sample_id = self.sample_ids[index]
        data_dir = os.path.join(self.data_root, sample_id)
        label_file = os.path.join(self.label_root, f"{sample_id}.mat")
        
        # 1. Load Images
        img_list = natsort.natsorted(os.listdir(data_dir))
        img_list = [x for x in img_list if x.endswith('.bmp')]
        
        if not img_list:
             # Handle empty directory or no bmps
             print(f"Warning: No images found for {sample_id}")
             # Return empty tensors or raise error? Raising error is safer.
             raise RuntimeError(f"No .bmp images found in {data_dir}")

        # Read first image to get dimensions
        first_img_path = os.path.join(data_dir, img_list[0])
        first_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
        depth, width = first_img.shape
        num_frames = len(img_list)
        
        # Initialize volume
        volume = np.zeros((num_frames, depth, width), dtype=np.float32)
        
        for i, img_name in enumerate(img_list):
            img_path = os.path.join(data_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            volume[i, :, :] = img
            
        # 2. Generate Segmentation Mask
        layers_data = scipy.io.loadmat(label_file)['Layer']
        
        # Check consistency
        if layers_data.shape[1] != num_frames:
             print(f"Warning: Mismatch in frames for {sample_id}. Images: {num_frames}, Layers: {layers_data.shape[1]}")
             # We can truncate or pad? For now, let's assume we proceed but it might crash if dimensions don't match broadcasting.
             # Actually, if shapes don't match, broadcasting below will fail or produce wrong result.
             # Let's trust the data for now or raise error.
        
        mask = np.zeros((num_frames, depth, width), dtype=np.int64)
        y_grid = np.arange(depth).reshape(1, depth, 1)
        
        for i in range(layers_data.shape[0]):
            layer_surface = layers_data[i, :, :]
            # Ensure layer_surface matches num_frames
            if layer_surface.shape[0] != num_frames:
                 # Resize or slice?
                 # If layer has more frames, slice. If fewer, error.
                 if layer_surface.shape[0] > num_frames:
                     layer_surface = layer_surface[:num_frames, :]
                 else:
                     raise RuntimeError(f"Not enough layer frames for {sample_id}")

            layer_surface_expanded = layer_surface[:, np.newaxis, :]
            mask += (y_grid > layer_surface_expanded).astype(np.int64)
            
        data_tensor = torch.from_numpy(volume).unsqueeze(0) # (1, Frames, Depth, Width)
        label_tensor = torch.from_numpy(mask) # (Frames, Depth, Width)
        
        return data_tensor, label_tensor
