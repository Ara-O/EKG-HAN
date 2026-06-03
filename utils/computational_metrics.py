from thop import profile, clever_format
import time
import numpy as np
import torch
import calflops

def analyze_model_complexity(model, input_size=(1, 10, 300, 12), device='cuda'):
    print("="*80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*80)

    model = model.to(device)
    model.eval()

    # 1. Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"\nParameter Count:")
    print(f"   Total Parameters:       {total_params:,}")
    print(f"   Trainable Parameters:   {trainable_params:,}")
    print(f"   Non-trainable Parameters: {non_trainable_params:,}")
    print(f"   Model Size (MB):        {total_params * 4 / (1024**2):.2f}")  # 4 bytes per float32

    # 2. FLOPs calculation
    dummy_input = torch.randn(input_size).to(device)

    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"\n FLOPs (Floating Point Operations):")
        print(f"   MACs (Multiply-Accumulate): {macs}")
        print(f"   Parameters (thop):          {params}")
    except Exception as e:
        print(f"\n FLOPs calculation failed: {e}")

    # 3. Layer-wise parameter breakdown
    print(f"\nLayer-wise Parameter Breakdown:")
    print(f"{'Layer Name':<40} {'Parameters':>15} {'% of Total':>12}")
    print("-"*70)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            percentage = 100 * param_count / trainable_params
            print(f"{name:<40} {param_count:>15,} {percentage:>11.2f}%")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024**2)
    }


def setup_benchmark_env():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def time_forward(model, inputs, device='cuda'):
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(inputs)
        end.record()

        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0  # seconds
    else:
        t0 = time.time()
        _ = model(inputs)
        return time.time() - t0


def benchmark_latency(
    model,
    input_shape,
    device='cuda',
    runs=100,
    warmup=20
):
    model = model.to(device).eval()

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            x = torch.randn(input_shape, device=device)
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            x = torch.randn(input_shape, device=device)
            elapsed = time_forward(model, x, device)
            times.append(elapsed)

    times = np.array(times)

    return {
        "mean_ms": times.mean() * 1000,
        "std_ms": times.std() * 1000,
        "median_ms": np.median(times) * 1000,
        "min_ms": times.min() * 1000,
        "max_ms": times.max() * 1000,
    }


def benchmark_throughput(
    model,
    input_shape,
    batch_size,
    device='cuda',
    runs=100,
    warmup=20
):
    model = model.to(device).eval()
    shape = (batch_size,) + input_shape[1:]

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            x = torch.randn(shape, device=device)
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            x = torch.randn(shape, device=device)
            elapsed = time_forward(model, x, device)
            times.append(elapsed)

    times = np.array(times)

    return {
        "batch_time_ms": times.mean() * 1000,
        "per_sample_ms": (times.mean() / batch_size) * 1000,
        "samples_per_sec": batch_size / times.mean()
    }

def run_inference_benchmark(
    model,
    input_shape=(1, 10, 300, 12),
    batch_size=16,
    device='cuda'
):
    setup_benchmark_env()

    print("=" * 80)
    print("INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Input shape: {input_shape}")
    print(f"Throughput batch size: {batch_size}")

    latency = benchmark_latency(
        model=model,
        input_shape=input_shape,
        device=device
    )

    throughput = benchmark_throughput(
        model=model,
        input_shape=input_shape,
        batch_size=batch_size,
        device=device
    )

    print("\nBatch-1 Latency (ms):")
    for k, v in latency.items():
        print(f"  {k:>10}: {v:.3f}")

    print("\nBatch-N Throughput:")
    for k, v in throughput.items():
        if "ms" in k:
            print(f"  {k:>16}: {v:.3f}")
        else:
            print(f"  {k:>16}: {v:.2f}")

    return {
        "latency": latency,
        "throughput": throughput
    }



def calculate_flops(model):
    flops, macs, params = calflops.calculate_flops(model=model,
                                        input_shape=(1, 10, 300, 12),
                                        output_as_string=True,
                                        output_precision=4)
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))