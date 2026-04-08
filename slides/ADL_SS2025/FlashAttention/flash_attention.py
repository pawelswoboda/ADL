# %% [markdown]
## Flash Attention
#- Papers available here for [v1](https://arxiv.org/abs/2205.14135) and [v2](https://arxiv.org/abs/2307.08691)
#- Lecture based on excellent [video](https://www.youtube.com/watch?v=zy8ChVd_oTM) from Umar Jamil.
##### Problem with vanilla attention:
#- We need to instantiate $T \times T$ matrices for holding attention weights
#  - Memory consumption for long sequences.
#  - Interestingly, this is not an arithmetic problem: Computing attention scores is not the problem: <img src="./FlashAttention/flash_attention_time_ablation.png" width="300"/>

#- Closer look: HBM vs SRAM: <img src="./FlashAttention/hbm_vs_sram.png" width="300"/>
#  - Problem with vanilla attention is that after matmul we need to transfer attention matrix to HBM.
#  - Vanilla attention is memory-bound, not compute-bound.
#- Backward pass: backprop through multiple kernels, no automatic kernel fusion is performed.
#  - Store intermediate results (i.e. attention weights, ...) in memory.
#
#<img src="./FlashAttention/std_attention_implementation.png" width="300"/>

# %% [markdown]
##### Flash Attention:
#- Do not materialize the attention matrix.
#  - Compute attention weights on the fly (since is not the bottleneck)
#  - Use streaming softmax.
#  - In backward pass we will recompute attention weights as needed.
#- Memory-bound -> compute-bound (this is better).
#- Kernel-fusion: No additional nodes in the computational graph.
#
#<img src="./FlashAttention/flash_attention_illustration.png" width="300"/>

# %% [markdown]
##### Outline:
# We will implement FlashAttention v2 in triton.
#- GPU computational model, simple cuda examples.
#- Discussion of Flash Attention algorithm abstractly.
#- Implementation of forward pass in triton.
#- Derivation of backward pass.
#- Implementation of backward pass in triton.

# %% [markdown]
##### Softmax
#First, we will discuss how softmax is computed safely and efficiently in flash attention (and not only there).
#
#Softmax: $$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{i} e^{x_i}}$$
#Computational Efficiency: two passes over the data.
#- First: compute sum of exponentials
#- Second: Compute exponentials and divide by sum
#Problem: Numerically unstable for large $x$

# %%
#Softmax:

import math

def softmax(x):
    l = sum([math.exp(v) for v in x])
    return [math.exp(v) / l for v in x]

x = [1,2,3]
print(softmax([1,2,3]))
print(softmax([100,200,300]))
# print(softmax([1000,2000,3000])) # will overflow

# %% [markdown]
##### Stable Softmax:
#Solution: Subtract max from $x$ before applying softmax.
#$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{i} e^{x_i - \max(x)}}$$
#Computational Efficiency: three passes over the data.
#- First: find max
#- Second: compute sum of exponentials
#- Third: Compute exponentials and divide by sum

# %%
#Stable softmax:

def stable_softmax(x):
    m = max(x)
    l = sum([math.exp(v - m) for v in x])
    return [math.exp(v - m) / l for v in x]

print(stable_softmax([1,2,3]))
print(stable_softmax([100,200,300]))
print(stable_softmax([1000,2000,3000])) # will work now



# %% [markdown]
##### Streaming Softmax:
#Reduce one pass over the data: Fuse computation of max and normalization into one pass.

# %%
#Streaming softmax:

from functools import reduce

def streaming_softmax(x):
    m = float('-inf')
    l = 0
    for v in x:
        m_next = max(m, v)
        l = l * math.exp(m - m_next) + math.exp(v - m_next)
        m = m_next
    return [math.exp(v - m)/l for v in x]

print(streaming_softmax([1,2,3]))
print(streaming_softmax([100,200,300]))
print(streaming_softmax([1000,2000,3000])) # will work now

# %% [markdown]
## Cuda:
#- Cuda is the programming model for NVIDIA GPUs.
#
#<img src="./FlashAttention/CPU_vs_GPU2.png" width="500"/>
#
#- Model: SIMT: Single Instruction, Multiple Threads.
#- Threads are organized into blocks.
#- Blocks are organized into grids.
#- Cuda kernels are invoked with special syntax `<<<blocks, threads>>>`.

# %% [markdown]
#Vector addition example:
#```text
#__global__ void cuda_vector_add_simple(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int N)
#{
#    int i = threadIdx.x;
#    if (i < N)
#    {
#        OUT[i] = A[i] + B[i];
#    }
#}
#
#int N = 1024;
#int *d_A, *d_B, *d_OUT;
#cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * N);
#cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * N);
#cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * N);
#cuda_vector_add_simple<<<1, N>>>(d_OUT, d_A, d_B, N);
#```

# %% [markdown]
#Matrix addition example:
#```text
#__global__ void cuda_matrix_add(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int NUM_ROWS, int NUM_COLS)
#{
#    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
#    int col_index = blockIdx.x * blockDim.x + threadIdx.x;
#
#    if (row_index < NUM_ROWS && col_index < NUM_COLS)
#    {
#        size_t index = static_cast<size_t>(row_index) * NUM_COLS + col_index; // A[row_index][col_index]
#        OUT[index] = A[index] + B[index];
#    }
#}
#
#int NUM_COLS = 1024;
#int NUM_ROWS = 512;
#int *d_A, *d_B, *d_OUT;
#cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * NUM_COLS * NUM_ROWS);
#cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * NUM_COLS * NUM_ROWS);
#cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * NUM_COLS * NUM_ROWS);
#
#int num_blocks_ROWS = ceil(NUM_ROWS / ROWS_block_size)
#int num_blocks_COLS = ceil(NUM_COLS / COLS_block_size)
#
#dim3 grid(num_blocks_COLS, num_blocks_ROWS, 1);
#dim3 block(COLS_block_size, ROWS_block_size, 1);
#
#cuda_matrix_add<<<grid, block>>>(d_OUT, d_A, d_B, NUM_ROWS, NUM_COLS);
#```

# %% [markdown]
## General Idea:
#- Blocks: Collection of thread blocks
#  - Indexed by `blockIdx.x`, `blockIdx.y`, `blockIdx.z`.
#- Threads: Group of threads that execute the same code.
#  - Up to 1024 threads ber block.
#  - Indexed by `threadIdx.x`, `threadIdx.y`, `threadIdx.z`.
#  - Runs on a single streaming multiprocessor (SM)
#  - Warp is a group of 32 threads that execute instructions in lockstep.

# %% [markdown]
## Triton:
#- Triton is a python programming language extension and compiler for writing GPU kernels.
#- Abstract away some low-level details of CUDA.
#- Similar concept of blocks and threads:
#  - "programs" or "tasks" instead of "blocks".
#  - Threads are not directly exposed.

# %%
#Triton vector addition example:
#- BLOCK_SIZE: Number of threads in a block.
#  - is annotated with `@tl.constexpr`.
#  - Is passed via the special meta dictionary
#- pid: Program ID (block ID).
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(output_ptr + offsets, result, mask=mask)

# Example usage
n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
output = torch.empty_like(x)

BLOCK_SIZE = 256
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)

# Validate result
assert torch.allclose(output, x + y)

# %%
#Triton matrix addition example:
import torch
import triton
import triton.language as tl

@triton.jit
def mat_add_kernel(A, B, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    A_block = tl.load(A + offs_m[:, None] * N + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :])
    B_block = tl.load(B + offs_m[:, None] * N + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :])
    C_block = A_block + B_block
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], C_block, mask=mask_m[:, None] & mask_n[None, :])

# Example usage
M, N = 512, 512
A = torch.randn((M, N), device='cuda')
B = torch.randn((M, N), device='cuda')
C = torch.empty((M, N), device='cuda')

BLOCK_M = 64
BLOCK_N = 64
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
mat_add_kernel[grid](A, B, C, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

# Validate result
assert torch.allclose(C, A + B)

# %% [markdown]
##Blockwise Matrix Multiplication
#We will later, for calculating the attention weights, need to perform matrix multiplication.
#To this end we will see how to do it in block-wise manner and implement this in triton.
#
#<img src="./FlashAttention/block_matrix_multiplication.png" width="600"/>

# %% [markdown]
## Tensor Layout
#- In order to access elements in vectors/matrices and tensors, let us review how they are stored in memory.
#- Since we need to access them in triton without the convenient indexing of torch/numpy, this is essential to know.
#- Tensor information is stored in two arrays:
#  - Shape: Number of elements in each dimension.
#  - Strides are the number of elements to skip in each dimension to get to the next element.
#- Vectors:
#
#<img src="./FlashAttention/vector_memory_layout.png" width="600"/>
#
#- Matrices: Row-major layout:
#
#<img src="./FlashAttention/matrix_memory_layout.png" width="600"/>

# %%
#Block Matrix Multiplication in Triton

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr   #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
assert torch.allclose(triton_output, torch_output)


# %% [markdown]
## Fun with strides (for later use)
#- We can transpose a matrix without physically moving the data in memory.
#- We have to exchange the strides
#- But tensor is no longer contiguous.
#  - In torch you cannot use `view` but need to `reshape` on non-contiguous tensors.
#
#<img src="./FlashAttention/matrix_stride_transpose.png" width="600"/>

# %% [markdown]
# We finally have enough background to implement the forward pass of FlashAttention.
#- Recall the attention formula:
#
#<img src="./FlashAttention/attention_formula.png" width="300"/>
#
#- For simplicity consider the case of two blocks:
#
#<img src="./FlashAttention/attention_with_two_blocks_S.png" width="100"/>
#
#- This is the case for two values
#
#<img src="./FlashAttention/attention_with_two_blocks_V.png" width="70"/>
#
#- The attention formula in this case becomes:
#
#<img src="./FlashAttention/attention_with_two_blocks.png" width="400"/>
#
#- When using online softmax, we can incrementally compute the attention weights.
#
#<img src="./FlashAttention/attention_with_two_blocks_online_softmax.png" width="400"/>
#
#- Two small efficiency tricks:
# 
#<img src="./FlashAttention/O_scaling_trick.png" width="400"/>
#
#<img src="./FlashAttention/L_saving_trick.png" width="100"/>
#
#- The full algorithm is:
#
#<img src="./FlashAttention/flash_attention_2_forward_pass.png" width="400"/>


# %%
#Flash Attention v2 forward pass in Triton
@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q) # This lets the compiler know that lo is a multiple of BLOCK_SIZE_Q, used for optimizations
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the (N_CTX, HEAD_DIM) block in the Q, K, V by selecting indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)

test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=256, HEAD_DIM=32, causal=False)

# %%
