# Triton vector addition: OUT[i] = A[i] + B[i]
#
# One Triton program per BLOCK_SIZE-wide tile. Each program loads two tiles
# from HBM, adds them, and stores one tile back. Like the CUDA version, this
# kernel is memory-bound: peak throughput is the HBM bandwidth.
#
# In CUDA, each thread handles one element and we pick a block size so that
# warps coalesce. In Triton, the unit of work is the whole tile; the compiler
# vectorises the loads and stores within each tile for us.
#
# Run: python vector_add.py

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(OUT, A, B, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    tl.store(OUT + offsets, a + b, mask=mask)


def vector_add(a, b, BLOCK_SIZE=1024):
    n = a.numel()
    out = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](out, a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def check_correctness():
    N = 1 << 16
    torch.manual_seed(0)
    a = torch.rand(N, device="cuda")
    b = torch.rand(N, device="cuda")
    out = vector_add(a, b)
    ref = a + b
    err = (out - ref).abs().max().item()
    ok = err < 1e-5
    print(f"[correctness] N={N}  max abs err={err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def benchmark():
    sizes = [1 << 16, 1 << 20, 1 << 24, 1 << 26]
    print(f"\n{'N':<10} {'ms/iter':>12} {'GFLOP/s':>10} {'GB/s':>10}")
    for N in sizes:
        a = torch.zeros(N, device="cuda")
        b = torch.zeros(N, device="cuda")
        ms = triton.testing.do_bench(lambda: vector_add(a, b))
        gflops = N / (ms * 1e6)
        gbs = N * a.element_size() * 3 / (ms * 1e6)
        print(f"{N:<10} {ms:>12.5f} {gflops:>10.2f} {gbs:>10.2f}")


if __name__ == "__main__":
    if not check_correctness():
        raise SystemExit(1)
    benchmark()
