import argparse
import sys
import torch
from pathlib import Path

# Add project root to path if needed (though usually running from root works)
sys.path.append(str(Path(__file__).parent))

from src.utils.profiling import BenchmarkConfig, ModelFactory, Profiler

def parse_args():
    parser = argparse.ArgumentParser(description="Unified NCA Benchmark Tool")
    
    # Model Config
    parser.add_argument("--model", type=str, default="fastseg", choices=["fastseg", "triton"], help="Model architecture")
    parser.add_argument("--steps", type=int, default=16, help="Number of NCA steps")
    parser.add_argument("--channels", type=int, default=32, help="Number of channels")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden size")
    
    # Input Config
    parser.add_argument("--size", type=int, nargs='+', default=[128, 128], help="Input size: H W (or single int for square)")
    
    # Optimization
    parser.add_argument("--compile", type=str, default="none", 
                        choices=["none", "default", "reduce-overhead", "max-autotune"], 
                        help="torch.compile mode")
    
    # Execution
    parser.add_argument("--mode", type=str, default="latency", choices=["latency", "trace", "grid"], help="Benchmark mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Measurement runs")
    
    parser.add_argument("--config", type=str, help="Path to YAML config file (overrides CLI args)")
    
    return parser.parse_args()

def run_grid_search(args):
    """
    Example grid search over compile modes.
    """
    print("=== GRID SEARCH MODE ===")
    modes = ["none", "reduce-overhead", "max-autotune"]
    
    results = {}
    
    for m in modes:
        print(f"\n--- Testing Mode: {m} ---")
        # Create config override
        config = BenchmarkConfig(
            model_type=args.model,
            channel_n=args.channels,
            hidden_size=args.hidden,
            steps=args.steps,
            input_h=args.size[0] if isinstance(args.size, list) else args.size,
            input_w=args.size[1] if isinstance(args.size, list) else args.size,
            compile_mode=m,
            device=args.device if torch.cuda.is_available() else "cpu"
        )
        
        try:
            model = ModelFactory.create_model(config)
            profiler = Profiler(model, config)
            lat = profiler.benchmark_latency(n_warmup=10, n_runs=20) # Quick run
            results[m] = lat
        except Exception as e:
            print(f"Failed for {m}: {e}")
            results[m] = float('inf')
            
    print("\n=== Grid Results ===")
    for m, lat in results.items():
        print(f"{m:<15}: {lat:.2f} ms ({1000/lat:.1f} FPS)")

def main():
    args = parse_args()
    
    if args.config:
        print(f"Loading config from {args.config}...")
        config = BenchmarkConfig.from_yaml(args.config)
        # Allow CLI override if needed? For now, let's say config file is source of truth if provided
    else:
        # Handle Input Size normalization
        if isinstance(args.size, int):
            h, w = args.size, args.size
        elif len(args.size) == 1:
            h, w = args.size[0], args.size[0]
        else:
            h, w = args.size[0], args.size[1]
            
        config = BenchmarkConfig(
            model_type=args.model,
            channel_n=args.channels,
            hidden_size=args.hidden,
            steps=args.steps,
            input_h=h,
            input_w=w,
            compile_mode=args.compile,
            device=args.device if torch.cuda.is_available() else "cpu"
        )
        
    if args.mode == "grid":
        # Grid search logic needs to be adapted if config is passed, 
        # but for now let's keep grid search driven by args or allow it to mutate config
        # Simply reusing run_grid_search with args for now as grid search usually implies CLI driven exploration
        if args.config:
            # If config file provided, maybe we just want grid over that config?
            # Complexity: Grid search modifies config. 
            pass
        run_grid_search(args)
        return

    # 1. Setup
    print(f"Configuration: {config}")
    model = ModelFactory.create_model(config)
    profiler = Profiler(model, config)
    
    # 2. Run
    if args.mode == "latency":
        results = profiler.benchmark_latency(n_warmup=args.warmup, n_runs=args.runs)
        profiler.save_experiment(results)
    elif args.mode == "trace":
        profiler.export_trace()

if __name__ == "__main__":
    main()
