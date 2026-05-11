# Triton matrix addition: OUT[r,c] = A[r,c] + B[r,c]
#
# A single 2D kernel: each Triton program computes one BLOCK_M x BLOCK_N tile
# of the output. The compiler vectorises the block load over the contiguous
# axis automatically.
#
# Note on coalesced vs strided: in the CUDA version we wrote two kernels with
# swapped (row, col) -> (threadIdx.x, threadIdx.y) mappings to isolate the
# cost of non-coalesced HBM access. Triton hides that choice -- block loads
# are coalesced as long as the innermost block axis matches the contiguous
# tensor axis. To reproduce the "strided" case, pass a transposed view: same
# kernel, but the strides flip and each row of the block fetches values that
# are `cols` floats apart. The benchmark below does both.
#
# Run: python matrix_add.py

import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(OUT, A, B, M, N,
                      stride_am, stride_an,
                      stride_bm, stride_bn,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a_ptr = A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptr = B + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    o_ptr = OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    a = tl.load(a_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)
    tl.store(o_ptr, a + b, mask=mask)


def matrix_add(a, b, BLOCK_M=32, BLOCK_N=32):
    assert a.shape == b.shape
    M, N = a.shape
    out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))
    matrix_add_kernel[grid](
        out, a, b, M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def check_correctness():
    R, C = 257, 513  # deliberately not multiples of any block size
    torch.manual_seed(0)
    a = torch.rand(R, C, device="cuda")
    b = torch.rand(R, C, device="cuda")
    out = matrix_add(a, b)
    ref = a + b
    err = (out - ref).abs().max().item()
    ok = err < 1e-5
    print(f"[correctness] {R}x{C}  max abs err={err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def benchmark():
    R, C = 4096, 4096
    a = torch.zeros(R, C, device="cuda")
    b = torch.zeros(R, C, device="cuda")
    bytes_ = R * C * a.element_size() * 3

    # Transposed views: same data, swapped strides. The kernel issues the
    # same loads in logical-index space, but in HBM they step by R floats
    # along the innermost block axis -- the "strided" pattern.
    at, bt = a.t(), b.t()

    configs = [(32, 32), (64, 16), (16, 64), (128, 8), (8, 128), (256, 4), (4, 256)]
    print(f"\nBlock-shape sweep at {R}x{C}")
    print(f"{'block':<10} {'contig GB/s':>14} {'transposed GB/s':>18}")
    for BM, BN in configs:
        ms_c = triton.testing.do_bench(lambda: matrix_add(a, b, BLOCK_M=BM, BLOCK_N=BN))
        ms_t = triton.testing.do_bench(lambda: matrix_add(at, bt, BLOCK_M=BM, BLOCK_N=BN))
        label = f"{BM}x{BN}"
        print(f"{label:<10} {bytes_/(ms_c*1e6):>14.1f} {bytes_/(ms_t*1e6):>18.1f}")


if __name__ == "__main__":
    if not check_correctness():
        raise SystemExit(1)
    benchmark()
