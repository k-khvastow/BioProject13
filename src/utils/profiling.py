import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from torch.profiler import profile, ProfilerActivity, schedule

# Import Model Classes
from src.models.NCA import FastSegNCA, TritonSegNCA

@dataclass
class BenchmarkConfig:
    model_type: str = "fastseg" # 'fastseg' or 'triton'
    channel_n: int = 32
    hidden_size: int = 32
    steps: int = 16
    batch_size: int = 1
    input_h: int = 128
    input_w: int = 128
    compile_mode: str = "none" # 'none', 'reduce-overhead', 'max-autotune'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "inference_reports"
    
    # Training stats for normalization (dummy data)
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])

    @classmethod
    def from_yaml(cls, path: str):
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        # Filter keys to only those in dataclass
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

class ModelFactory:
    @staticmethod
    def create_model(config: BenchmarkConfig) -> nn.Module:
        print(f"Instantiating {config.model_type} model...")
        device = torch.device(config.device)
        
        if config.model_type.lower() == "fastseg":
            model = FastSegNCA(
                channel_n=config.channel_n,
                hidden_size=config.hidden_size,
                input_channels=3,
                device=config.device
            )
        elif config.model_type.lower() == "triton":
             model = TritonSegNCA(
                channel_n=config.channel_n,
                hidden_size=config.hidden_size,
                input_channels=3,
                device=config.device
            )
             # Triton specific init
             model.cache_weights()
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
        model.to(device)
        model.eval()
        
        # Optimization
        if config.compile_mode != "none":
            print(f"Applying torch.compile(mode='{config.compile_mode}')...")
            # For Triton models, compilation might conflict or be redundant, 
            # but allow it if user requests.
            model = torch.compile(model, mode=config.compile_mode)
            
        return model

class InputFactory:
    @staticmethod
    def get_dummy_input(config: BenchmarkConfig) -> torch.Tensor:
        """
        Creates a dummy seed tensor (Batch, Channel, H, W).
        Channels 0-3 are initialized with random "image" data.
        """
        device = torch.device(config.device)
        
        # 1. Random Image (B, 3, H, W)
        img = torch.rand(config.batch_size, 3, config.input_h, config.input_w, device=device)
        
        # 2. Full State
        state = torch.zeros(
            config.batch_size, 
            config.channel_n, 
            config.input_h, 
            config.input_w, 
            device=device
        )
        
        # 3. Embed Image
        state[:, :3, :, :] = img
        
        return state

class Profiler:
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
    def benchmark_latency(self, n_warmup=20, n_runs=100) -> dict:
        """
        Runs latency benchmark.
        Returns: Dictionary with metrics.
        """
        print(f"Benchmarking Latency: {self.config.input_h}x{self.config.input_w}, {self.config.steps} steps")
        input_tensor = InputFactory.get_dummy_input(self.config)
        
        # Warmup
        print(f"Warmup ({n_warmup} runs)...")
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(input_tensor, steps=self.config.steps)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Timing
        print(f"Measuring ({n_runs} runs)...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    start_event.record()
                    
                    _ = self.model(input_tensor, steps=self.config.steps)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    timings.append(start_event.elapsed_time(end_event)) # Returns ms
                else:
                    # CPU Timing fallback
                    s = time.time()
                    _ = self.model(input_tensor, steps=self.config.steps)
                    e = time.time()
                    timings.append((e - s) * 1000)

        avg_lat = np.mean(timings)
        std_lat = np.std(timings)
        fps = 1000.0 / avg_lat
        
        results = {
            "latency_mean_ms": float(avg_lat),
            "latency_std_ms": float(std_lat),
            "fps": float(fps),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        print(f"Results: {avg_lat:.2f} ms ± {std_lat:.2f} | {fps:.2f} FPS")
        return results

    def save_experiment(self, results: dict):
        """
        Saves the experiment configuration and results to a timestamped folder.
        """
        import yaml
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{self.config.model_type}_{self.config.compile_mode}"
        exp_dir = Path(self.config.output_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Config (YAML)
        config_dict = self.config.__dict__.copy()
        # Remove fields that are not serializable or irrelevant if any
        if 'mean' in config_dict: del config_dict['mean']
        if 'std' in config_dict: del config_dict['std']
        
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
            
        # 2. Save Results (JSON)
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Experiment saved to: {exp_dir}")

    def export_trace(self):
        """
        Runs profiler and exports Chrome trace.
        """
        print("Profiling Kernels & Exporting Trace...")
        input_tensor = InputFactory.get_dummy_input(self.config)
        
        # Prepare Output Path
        out_dir = Path(self.config.output_dir) / "traces"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trace_{self.config.model_type}_{self.config.input_h}x{self.config.input_w}_{self.config.compile_mode}_{timestamp}.json"
        out_path = out_dir / filename
        
        def trace_handler(prof):
            print(f"Saving trace to {out_path}")
            prof.export_chrome_trace(str(out_path))
            
            # Also save summary table
            summary_path = out_path.with_suffix('.txt')
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            summary_path.write_text(table)
            print(f"Saved summary to {summary_path}")

        # Run Profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=3, active=5, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(1 + 3 + 5): # wait + warmup + active
                    _ = self.model(input_tensor, steps=self.config.steps)
                    prof.step()
