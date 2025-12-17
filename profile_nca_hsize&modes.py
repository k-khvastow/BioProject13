import torch
import os
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
import datetime

# Import your model definition
from src.models.NCA import FastSegNCA

# --- Configuration ---
#MODEL_PATH = Path("/vol/data/BioProject13/output_acevedo/Acevedo_FastSeg_nuclei_32epochs.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "channel_n": 128,
    "hidden_size": 128,
    "batch_size": 1,
    "steps": 64,                 # Number of NCA steps
    "input_size": 64,           # Resolution (Square) or set H/W below
    "input_h": 64,
    "input_w": 64,
    "compile_mode": "reduce-overhead", # 'none', 'default', 'reduce-overhead', 'max-autotune', 'min-autotune'
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
    hidden_size = config['hidden_size']
    h = config.get("input_h", config["input_size"])
    w = config.get("input_w", config["input_size"])
    steps = config["steps"]
    mode = config["compile_mode"]
    
    return f"{base_name}_hs:{hidden_size}_{h}x{w}_{steps}steps_{mode}_{timestamp}{extension}"
    #return f"{base_name}_hs:{config['hidden_size']}_{config.get("input_h", config["input_size"])}x{config.get("input_w", config["input_size"])}_{config["steps"]}steps_{config["compile_mode"]}_{timestamp}{extension}"

def load_model(config: dict) -> torch.nn.Module:
    print(f"Loading FastSegNCA model")# from {path}...")
    model = FastSegNCA(
        channel_n=config["channel_n"], 
        hidden_size=config["hidden_size"], 
        input_channels=3
    )    
        
    model.to(DEVICE)
    model.eval()
    
    # Apply Torch Compile based on Config
    mode = config["compile_mode"]
    print(f"Compiling model with mode='{mode}'...")
    if mode != "none":
        model = torch.compile(model, mode=mode)
    
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
    trace_dir = Path("./log/profiler/size&modes")
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    trace_filename = f"tracefile_{output_path.stem}.json"
    trace_file_path = trace_dir / trace_filename
    
    # 2. Define a custom handler to save the file with that name
    def custom_handler(prof):
        print(f"Exporting trace to")#: {trace_file_path.resolve()}")
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
    print(f"\nDetailed kernel table saved")# to: **{output_path.resolve()}**")

### testing hidden_size impact
if __name__ == "__main__":
    input_tensor = prepare_input(CONFIG)
    #hidden size testing loop
    dense_layer_size = [64, 128, 256, 512]
    compile_mode = ["reduce-overhead", 'none', 'default', 'max-autotune', 'max-autotune-no-cudagraphs']

    for size in dense_layer_size[1:]:
        for mode in compile_mode:
            print(f'---{size}, {mode}----')
            CONFIG["hidden_size"] = size
            CONFIG['compile_mode'] = mode

            #model = FastSegNCA(channel_n=CONFIG["channel_n"], hidden_size=CONFIG["hidden_size"])
            model = load_model(CONFIG)
            # benchmark latency
            benchmark_latency(model, input_tensor)

            base_name = f"NCA_Profile_"
            filename = get_detailed_filename(base_name, config=CONFIG)
            output_file_path = Path("./inference_reports/size&modes") / filename
            output_file_path.parent.mkdir(exist_ok=True)
            
            # 3. Run profiler and save output
            profile_kernels(model, input_tensor, output_file_path)