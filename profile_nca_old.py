import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
import datetime

# Import your model definition
from src.models.NCA import SegNCA

# --- Configuration ---
MODEL_PATH = Path("/vol/data/BioProject13/output_acevedo/Acevedo_Seg_nuclei_32epochs.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "channel_n": 128,
    "hidden_size": 128,
    "steps": 64,      
    "input_size": 64, 
    "batch_size": 1,
    # Stats from your training logic
    "mean": [0.82069695, 0.7281261, 0.836143],
    "std":  [0.16157213, 0.2490039, 0.09052657]
}

def get_timestamped_filename(base_name: str, extension: str = ".txt") -> str:
    """Creates a unique filename using the current date and time."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}{extension}"

def load_model(path: Path, config: dict) -> torch.nn.Module:
    print(f"Loading model from {path}...")
    model = SegNCA(
        channel_n=config["channel_n"], 
        hidden_size=config["hidden_size"], 
        input_channels=3
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def prepare_input(config):
    """
    Replicates the exact logic from SegmentationDataset:
    1. Create/Load Image
    2. Normalize (Mean/Std)
    3. Permute to (H, W, C)
    4. Embed into Seed State (H, W, 128)
    """
    # 1. Simulate a loaded image (H, W, 3) with values 0-255
    # In a real app, you would do: img = Image.open("path").convert("RGB")
    dummy_img_np = np.random.randint(0, 255, (config["input_size"], config["input_size"], 3), dtype=np.uint8)

    # 2. Define Transform (Exact same as your Dataset)
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=config["mean"], std=config["std"])
    ])

    # Apply Transform -> Output is (C, H, W)
    img_tensor = transform(dummy_img_np)

    # 3. Permute to (H, W, C) matches your Dataset __getitem__ return
    img_tensor = img_tensor.permute(1, 2, 0)
    
    # 4. Create the full Seed State (Batch, H, W, 128)
    seed = torch.zeros(
        config["batch_size"], 
        config["input_size"], 
        config["input_size"], 
        config["channel_n"]
    )
    
    # Embed the image into the first 3 channels
    seed[0, :, :, :3] = img_tensor
    
    return seed.to(DEVICE)

def benchmark_latency(model, input_tensor, n_runs=50):
    print(f"\n--- Benchmarking Latency ({n_runs} runs) ---")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor, steps=CONFIG["steps"])
    
    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            start_event.record()
            
            _ = model(input_tensor, steps=CONFIG["steps"])
            
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
            
    print(f"Average Inference Time: {np.mean(timings):.2f} ms")
    print(f"FPS: {1000 / np.mean(timings):.2f}")

def profile_kernels(model, input_tensor, output_path: Path):
    """
    Uses PyTorch Profiler to analyze specific CUDA kernels and memory usage, 
    and saves the output to a text file.
    """
    print("\n--- Profiling Kernels ---")
    
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor, steps=CONFIG["steps"])
                prof.step()

    # Capture the detailed table output
    kernel_summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    
    # Save the output to the specified file
    output_path.write_text(kernel_summary)
    
    print(f"\nDetailed kernel table saved to: **{output_path.resolve()}**")
    print("Detailed trace saved to ./log/profiler for visualization in TensorBoard.")

if __name__ == "__main__":
    # ... (Model and input tensor creation code)
    model = load_model(MODEL_PATH, CONFIG)
    input_tensor = prepare_input(CONFIG)
    
    # 1. Benchmark latency (optional: capture and save this output as well)
    benchmark_latency(model, input_tensor)
    
    # 2. Generate the unique filename
    base_name = f"NCA_Profile_{CONFIG['input_size']}x{CONFIG['input_size']}"
    filename = get_timestamped_filename(base_name)
    output_file_path = Path("./inference_reports") / filename
    
    # Ensure the directory exists
    output_file_path.parent.mkdir(exist_ok=True)
    
    # 3. Run profiler and save output
    profile_kernels(model, input_tensor, output_file_path)