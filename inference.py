"""
Inference script to apply a trained FastSegNCA model to a single image.
Loads a checkpoint and generates segmentation predictions.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2

# Project specific imports
from src.models.NCA import FastSegNCA


def load_model(checkpoint_path: str, channel_n: int = 128, hidden_size: int = 128, input_channels: int = 3, device: str = "cuda") -> FastSegNCA:
    """
    Load a trained FastSegNCA model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        channel_n: Number of channels in the model
        hidden_size: Hidden layer size
        input_channels: Number of input channels (3 for RGB)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        FastSegNCA model with loaded weights
    """
    model = FastSegNCA(channel_n=channel_n, hidden_size=hidden_size, input_channels=input_channels)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    return model


def preprocess_image(image_path: str, resize: int = 64, channel_n: int = 128, mean: list = None, std: list = None, device: str = "cuda") -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess a single image for inference, matching the training pipeline.
    
    Args:
        image_path: Path to the input image
        resize: Target size for resizing
        channel_n: Number of channels to pad to (for NCA seed)
        mean: Mean for normalization (if None, uses ImageNet defaults)
        std: Std for normalization (if None, uses ImageNet defaults)
        device: Device to move tensor to ('cuda' or 'cpu')
    
    Returns:
        Tuple of (preprocessed tensor in (B, C, H, W) format, original PIL image)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((resize, resize))
    img_np = np.array(img_resized)
    
    # Transform
    transform = v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])
    
    img_tensor = transform(img_np)
    
    # Permute to (H, W, C) format
    img_tensor = img_tensor.permute(1, 2, 0)
    
    # Add batch dimension: (1, H, W, C)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Create seed by padding to channel_n (matches training preprocessing)
    seed = torch.zeros((img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2], channel_n), dtype=torch.float32, device=device)
    seed[..., :img_tensor.shape[-1]] = img_tensor.to(device)
    
    # Permute to (B, C, H, W) for FastSegNCA (matches TrainingAgent.prepare_data)
    seed = seed.permute(0, 3, 1, 2)
    
    print(f"Image preprocessed: {image_path}")
    print(f"  Final shape (B, C, H, W): {seed.shape}")
    
    return seed, img_np


def inference(model: FastSegNCA, img_tensor: torch.Tensor, steps: int = 64, device: str = "cuda") -> torch.Tensor:
    """
    Run inference on a single image.
    
    Args:
        model: Trained FastSegNCA model
        img_tensor: Preprocessed image tensor (1, H, W, C) on device
        steps: Number of NCA steps to run
        device: Device to run inference on
    
    Returns:
        Model output prediction
    """
    with torch.no_grad():
        # Forward pass through the model
        output, feature_map = model(img_tensor, steps=steps, fire_rate=0.5)
    
    print(f"Inference complete ({steps} steps)")
    print(f"  Output shape: {output.shape}")
    
    return output, feature_map


def postprocess_output(output: torch.Tensor) -> np.ndarray:
    """
    Post-process model output to get segmentation mask.
    
    Args:
        output: Model output tensor (B, H, W) from FastSegNCA
    
    Returns:
        Segmentation mask (H, W) with values 0-1
    """
    # Apply sigmoid to get probabilities
    pred_prob = torch.sigmoid(output)
    
    # Remove batch dimension
    pred_mask = pred_prob[0].cpu().numpy()
    
    print(f"Output post-processed")
    print(f"  Mask shape: {pred_mask.shape}")
    print(f"  Mask range: [{pred_mask.min():.4f}, {pred_mask.max():.4f}]")
    
    return pred_mask


def visualize_results(original_img: np.ndarray, pred_mask: np.ndarray, save_path: str = "segmentation_result.png") -> None:
    """
    Visualize original image and segmentation prediction.
    
    Args:
        original_img: Original input image (H, W, C) normalized to [0, 1]
        pred_mask: Prediction mask (H, W)
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Prediction heatmap
    axes[1].imshow(pred_mask, cmap="hot")
    axes[1].set_title("Prediction (Probability)")
    axes[1].axis("off")
    cbar1 = plt.colorbar(axes[1].images[0], ax=axes[1])
    cbar1.set_label("Probability")
    
    # Binary mask (threshold at 0.5)
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title("Binary Mask (threshold=0.5)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {save_path}")
    plt.close()


def run_inference_on_image(
    image_path: str,
    checkpoint_path: str,
    output_dir: str = "inference_results",
    resize: int = 64,
    steps: int = 64,
    channel_n: int = 128,
    hidden_size: int = 128,
    mean: list = None,
    std: list = None,
):
    """
    Complete inference pipeline: load model, preprocess image, run inference, visualize.
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save results
        resize: Image resize dimension
        steps: Number of NCA steps
        channel_n: Number of channels in model
        hidden_size: Hidden layer size
        mean: Normalization mean
        std: Normalization std
    """
    print("=" * 70)
    print("FastSegNCA Inference Pipeline")
    print("=" * 70)
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")
    
    # Load model
    model = load_model(checkpoint_path, channel_n=channel_n, hidden_size=hidden_size, device=device)
    
    # Preprocess
    img_tensor, img_np = preprocess_image(image_path, resize=resize, channel_n=channel_n, mean=mean, std=std, device=device)
    
    # Inference
    output, feature_map = inference(model, img_tensor, steps=steps, device=device)
    
    # Post-process
    pred_mask = postprocess_output(output)
    
    # Save prediction mask
    mask_path = os.path.join(output_dir, "segmentation_mask.npy")
    np.save(mask_path, pred_mask)
    print(f"Mask saved to: {mask_path}")
    
    # Visualize
    viz_path = os.path.join(output_dir, "segmentation_visualization.png")
    visualize_results(img_np, pred_mask, viz_path)
    
    # Save binary mask as PNG
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(binary_mask, mode="L").save(os.path.join(output_dir, "segmentation_binary.png"))
    print(f"Binary mask saved to: {os.path.join(output_dir, 'segmentation_binary.png')}")
    
    print("\n" + "=" * 70)
    print("Inference complete!")
    print("=" * 70)
    
    return pred_mask


if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "/home/ubuntu/BioProject13/output_acevedo/Acevedo_FastSeg_nuclei_32epochs.pth"
    IMAGE_PATH = "/home/ubuntu/BioProject13/acevedo/images/basophil/BA_6109.jpg"
    OUTPUT_DIR = "/home/ubuntu/BioProject13/inference_results"
    """
    CHECKPOINT_PATH = "/vol/data/BioProject13/output_acevedo/Acevedo_FastSeg_nuclei_32epochs.pth"
    IMAGE_PATH = "/vol/data/BioProject13/acevedo/images/basophil/BA_6109.jpg"  # Change to your test image
    OUTPUT_DIR = "/vol/data/BioProject13/inference_results"
    """
    # Optional: specify your own normalization statistics
    # MEAN = [0.485, 0.456, 0.406]
    # STD = [0.229, 0.224, 0.225]
    MEAN = None
    STD = None
    
    # Run inference
    pred_mask = run_inference_on_image(
        image_path=IMAGE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        resize=64,
        steps=64,
        channel_n=128,
        hidden_size=128,
        mean=MEAN,
        std=STD,
    )
