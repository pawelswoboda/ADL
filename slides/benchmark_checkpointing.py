"""
Benchmark gradient checkpointing on GPT-2.
Compares memory usage and backward pass speed with and without checkpointing.

Usage: python benchmark_checkpointing.py [--device cuda|mps|cpu] [--seq-len 512] [--batch-size 4]
"""

import argparse
import time
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.checkpoint import checkpoint


def get_memory_mb(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1024**2
    elif device.type == "mps":
        # MPS doesn't expose fine-grained memory stats
        return torch.mps.current_allocated_memory() / 1024**2
    return 0.0


def reset_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def enable_checkpointing(model):
    """Wrap each transformer block with gradient checkpointing."""
    for block in model.transformer.h:
        block._original_forward = block.forward

        def make_ckpt_forward(module):
            def ckpt_forward(*args, **kwargs):
                return checkpoint(
                    module._original_forward, *args, use_reentrant=False, **kwargs
                )
            return ckpt_forward

        block.forward = make_ckpt_forward(block)


def disable_checkpointing(model):
    """Restore original forward methods."""
    for block in model.transformer.h:
        if hasattr(block, "_original_forward"):
            block.forward = block._original_forward
            del block._original_forward


def run_benchmark(model, input_ids, labels, device, n_warmup=2, n_runs=5):
    """Run forward+backward, return (peak_memory_mb, avg_time_s)."""
    # Warmup
    for _ in range(n_warmup):
        out = model(input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    reset_memory(device)

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = model(input_ids, labels=labels)
        out.loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        model.zero_grad(set_to_none=True)

    peak_mem = get_memory_mb(device)
    avg_time = sum(times) / len(times)
    return peak_mem, avg_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")

    if device.type == "cpu":
        print("Warning: memory tracking only works on CUDA/MPS. Times will still be measured.\n")

    # Load GPT-2 (124M)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M\n")

    # Dummy input
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()

    # --- Without checkpointing ---
    print("=" * 50)
    print("WITHOUT gradient checkpointing")
    print("=" * 50)
    reset_memory(device)
    mem_no_ckpt, time_no_ckpt = run_benchmark(model, input_ids, labels, device, n_runs=args.n_runs)
    print(f"  Peak memory:  {mem_no_ckpt:>8.1f} MB")
    print(f"  Avg fwd+bwd:  {time_no_ckpt*1000:>8.1f} ms")

    # --- With checkpointing ---
    print()
    print("=" * 50)
    print("WITH gradient checkpointing")
    print("=" * 50)
    model.zero_grad(set_to_none=True)
    enable_checkpointing(model)
    reset_memory(device)
    mem_ckpt, time_ckpt = run_benchmark(model, input_ids, labels, device, n_runs=args.n_runs)
    print(f"  Peak memory:  {mem_ckpt:>8.1f} MB")
    print(f"  Avg fwd+bwd:  {time_ckpt*1000:>8.1f} ms")

    # --- Comparison ---
    print()
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)
    if mem_no_ckpt > 0:
        mem_saved = mem_no_ckpt - mem_ckpt
        print(f"  Memory saved:   {mem_saved:>8.1f} MB ({mem_saved/mem_no_ckpt*100:.1f}%)")
    slowdown = (time_ckpt - time_no_ckpt) / time_no_ckpt * 100
    print(f"  Time overhead:  {slowdown:>+8.1f}%")

    disable_checkpointing(model)


if __name__ == "__main__":
    main()
