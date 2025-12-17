import torch
import os
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
import datetime

# Import your model definition
from src.models.NCA import TritonSegNCA

# --- Configuration ---
MODEL_PATH = Path("/vol/data/BioProject13/output_acevedo/Acevedo_FastSeg_nuclei_32epochs.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "channel_n": 128,
    "hidden_size": 128,
    "batch_size": 1,
    "steps": 64,                 # Number of NCA steps
    "input_size": 440,           # Resolution (Square) or set H/W below
    "input_h": 300,
    "input_w": 440,
    # Stats from your training logic
    "mean": [0.82069695, 0.7281261, 0.836143],
    "std":  [0.16157213, 0.2490039, 0.09052657]
}

def get_detailed_filename(base_name: str, config: dict, extension: str = ".txt") -> str:
    """
    Creates a filename like:
    NCA_Profile_440x300_64steps_reduce-overhead_2023-10-27_10-30-00.txt
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    h = config.get("input_h", config["input_size"])
    w = config.get("input_w", config["input_size"])
    steps = config["steps"]
    mode = config["compile_mode"]
    
    return f"{base_name}_{h}x{w}_{steps}steps_{mode}_{timestamp}{extension}"

def load_model(path: Path, config: dict) -> torch.nn.Module:
    print(f"Loading TritonSegNCA model from {path}...")
    model = TritonSegNCA(
        channel_n=config["channel_n"], 
        hidden_size=config["hidden_size"], 
        input_channels=3
    )
    
    # Try loading weights (handling potential mismatches if needed)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Warning: Exact key match failed, trying strict=False. Error: {e}")
        model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
        
    model.cache_weights()
    model.eval()
    
    return model

def prepare_input(config):
    """
    Prepares input in NCHW format (Batch, Channel, Height, Width)
    Compatible with FastSegNCA.
    """
    # 1. Determine size
    H = config.get("input_h", config["input_size"])
    W = config.get("input_w", config["input_size"])

    # 2. Create Dummy Image
    dummy_img_np = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # 3. Define Transform 
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=config["mean"], std=config["std"])
    ])

    # Apply Transform -> Output is (3, H, W)
    img_tensor = transform(dummy_img_np)

    # 4. Create the full Seed State in NCHW (Batch, Channel, H, W)
    # Crucial change: channels are dim 1, not dim 3
    seed = torch.zeros(
        config["batch_size"],
        config["channel_n"],
        H,
        W
    )

    # Embed the image into the first 3 channels
    seed[0, :3, :, :] = img_tensor
    
    return seed.to(DEVICE)

def benchmark_latency(model, input_tensor, n_runs=50):
    print(f"\n--- Benchmarking Latency ({n_runs} runs) ---")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(40):
            _ = model(input_tensor, steps=CONFIG["steps"])
    torch.cuda.synchronize()
    
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
    print("\n--- Profiling Kernels ---")
    
    # 1. Construct the exact trace filename you want
    # We use output_path.stem to get the filename WITHOUT the .txt extension
    trace_dir = Path("./log/profiler")
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    trace_filename = f"tracefile_{output_path.stem}.json"
    trace_file_path = trace_dir / trace_filename
    
    # 2. Define a custom handler to save the file with that name
    def custom_handler(prof):
        print(f"Exporting trace to: {trace_file_path.resolve()}")
        prof.export_chrome_trace(str(trace_file_path))

    # 3. Run Profiler
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=custom_handler,  # <--- Use our custom handler here
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor, steps=CONFIG["steps"])
                prof.step()

    # 4. Save the text summary as before
    kernel_summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    output_path.write_text(kernel_summary)
    print(f"\nDetailed kernel table saved to: **{output_path.resolve()}**")

if __name__ == "__main__":
    # ... (Model and input tensor creation code)
    model = load_model(MODEL_PATH, CONFIG)

    # Limit CPU threads for more consistent profiling
    # Ensures PyTorch doesn't spawn a thread pool for CPU operations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    # 1. Benchmark latency (optional: capture and save this output as well)
    #benchmark_latency(model, input_tensor)
    
    # Test multiple image sizes
    TEST_SIZES = [
        (64, 64),
        (128, 128),
        (256, 256),
        (440, 300),  # required size
    ]

    for H, W in TEST_SIZES:
        print(f"\n=== Benchmarking resolution: {H} x {W} ===")

        CONFIG["input_h"] = H
        CONFIG["input_w"] = W

        # Prepare dummy input of this size
        input_tensor = prepare_input(CONFIG)

        # Run benchmark
        benchmark_latency(model, input_tensor)

    # 2. Generate the unique filename
    filename = get_detailed_filename("Triton_Profile", CONFIG)
    output_file_path = Path("./inference_reports/triton_") / filename
    
    # Ensure the directory exists
    output_file_path.parent.mkdir(exist_ok=True)
    
    # 3. Run profiler and save output
    profile_kernels(model, input_tensor, output_file_path)