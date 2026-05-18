# Triton matrix multiplication: C = A * B
#   A is M x K, B is K x N, C is M x N (all row-major).
#
# A single tiled kernel: each Triton program computes one BLOCK_M x BLOCK_N
# block of C by iterating over the K dimension in BLOCK_K-wide chunks. Inside
# the loop, `tl.dot` performs a block-level matmul of one (BLOCK_M, BLOCK_K)
# x (BLOCK_K, BLOCK_N) chunk and accumulates into an fp32 register tile.
#
# Uses `tl.make_block_ptr` + `tl.advance` for indexing -- the same idiom used
# in the Flash Attention kernels below, so the two examples share style.
#
# Compared with the CUDA version:
#   * The CUDA `naive` kernel (one thread per output element) has no real
#     equivalent in Triton: the natural Triton kernel is already tiled.
#   * Triton stages each (A_tile, B_tile) chunk through SRAM and registers
#     automatically, so the explicit `__shared__` declarations and
#     `__syncthreads()` of the CUDA `tiled` kernel are not written here --
#     the compiler emits them.
#   * We benchmark against `torch.matmul` (cuBLAS) as the "gold standard".
#
# This is the same load-once / compute-many pattern Flash Attention applies
# to Q,K,V tiles for the attention matmul -- written here in its plain form.
#
# Run: python matmul.py

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block-pointer style: each tile carries its own base/shape/strides/offset.
    # `tl.advance` then walks the pointer along the K axis without re-computing
    # 2D pointer matrices, and `boundary_check` replaces manual masks.
    A_ptr = tl.make_block_ptr(
        base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    B_ptr = tl.make_block_ptr(
        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )
    C_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, K, BLOCK_K):
        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))  # walk A right along K
        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))  # walk B down along K

    tl.store(C_ptr, acc.to(C.dtype.element_ty), boundary_check=(0, 1))


def matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def check_correctness():
    M, N, K = 200, 250, 96  # not multiples of any block size
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    out = matmul(a, b)
    ref = a @ b
    err = (out - ref).abs().max().item()
    ok = err < 1e-3
    print(f"[correctness] {M}x{K} * {K}x{N}  max abs err={err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def benchmark():
    sizes = [256, 512, 1024, 2048]
    print(f"\n{'M=N=K':<8} {'triton ms':>12} {'triton GF/s':>14} "
          f"{'torch ms':>12} {'torch GF/s':>14} {'tri/torch':>10}")
    for S in sizes:
        a = torch.zeros(S, S, device="cuda")
        b = torch.zeros(S, S, device="cuda")
        ms_tr = triton.testing.do_bench(lambda: matmul(a, b))
        ms_to = triton.testing.do_bench(lambda: a @ b)
        gf_tr = 2.0 * S * S * S / (ms_tr * 1e6)
        gf_to = 2.0 * S * S * S / (ms_to * 1e6)
        print(f"{S:<8} {ms_tr:>12.5f} {gf_tr:>14.2f} "
              f"{ms_to:>12.5f} {gf_to:>14.2f} {ms_tr/ms_to:>9.2f}x")


if __name__ == "__main__":
    if not check_correctness():
        raise SystemExit(1)
    benchmark()
