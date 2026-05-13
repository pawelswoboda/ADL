import torch

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# STAGE conventions
# ---------------------------------------------------------------------------
# The outer kernel (`_attn_fwd`) receives a STAGE flag from the Python caller:
#   STAGE = 1  -> non-causal attention (no mask anywhere)
#   STAGE = 3  -> causal attention    (lower-triangular mask)
#
# Causal attention is split into two passes over the K/V sequence so that the
# expensive per-element masking only runs on the single diagonal block where
# masked and unmasked keys mix. The inner kernel (`_attn_fwd_inner`) takes its
# own STAGE flag, encoded as `4 - outer_STAGE` for the first call and `2` for
# the second:
#
#   inner STAGE = 1  -> "left of the diagonal" pass for causal attention.
#                       All keys in this range are fully attended to (no mask).
#                       Range: [0, block_index_q * BLOCK_SIZE_Q).
#   inner STAGE = 2  -> "on the diagonal" pass for causal attention.
#                       This is the one block where some keys are masked out;
#                       we apply the q >= k mask element-wise.
#                       Range: [block_index_q*BLOCK_SIZE_Q, (block_index_q+1)*BLOCK_SIZE_Q).
#   inner STAGE = 3  -> non-causal pass; iterate over the entire K/V sequence
#                       with no mask. Range: [0, SEQ_LEN).
#
# Splitting the causal case lets the unmasked-left pass run a tight, mask-free
# inner loop; only the diagonal block pays for the predicate evaluation.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Block sizes (tile sizes)
# ---------------------------------------------------------------------------
# The whole kernel is structured as a tiled matmul with on-the-fly softmax.
# We never materialize the full (SEQ_LEN x SEQ_LEN) attention matrix; instead
# each program computes a small rectangular tile of it at a time. The tile
# dimensions are the "block sizes." They are the central performance knob.
#
# Forward pass uses two block sizes:
#   BLOCK_SIZE_Q     -- number of query rows handled per program (per CTA).
#                       Each Triton program is launched once per Q block, so
#                       the launch grid is ceil(SEQ_LEN / BLOCK_SIZE_Q) along
#                       axis 0. Larger BLOCK_SIZE_Q means fewer programs,
#                       more work per program, more register pressure, and a
#                       larger Q tile that sits in registers/SRAM for the
#                       whole inner loop.
#   BLOCK_SIZE_KV    -- number of key/value rows loaded per inner-loop step.
#                       The inner loop streams the K/V sequence in chunks of
#                       this size. Larger BLOCK_SIZE_KV means fewer iterations
#                       and bigger matmul tiles (better tensor-core utilization)
#                       but more shared memory per program and worse occupancy.
#
# At each inner-loop step the kernel issues two matmuls:
#   QK^T :  (BLOCK_SIZE_Q,  HEAD_DIM)     @ (HEAD_DIM, BLOCK_SIZE_KV)
#   P  V :  (BLOCK_SIZE_Q,  BLOCK_SIZE_KV) @ (BLOCK_SIZE_KV, HEAD_DIM)
# So the on-chip tile footprint scales like
#   Q + K_tile + V_tile + acc
#   = BLOCK_SIZE_Q*HEAD_DIM + 2*BLOCK_SIZE_KV*HEAD_DIM + BLOCK_SIZE_Q*HEAD_DIM
# which must fit in registers + SRAM together with the running m_i / l_i.
#
# The forward kernel is wrapped in @triton.autotune over the cross product
#   BLOCK_SIZE_Q in {64, 128}, BLOCK_SIZE_KV in {32, 64}
# (plus num_warps and num_stages variations). Triton compiles every config,
# benchmarks them once per (SEQ_LEN, HEAD_DIM) pair, then caches the winner.
# A static_assert (BLOCK_SIZE_KV <= HEAD_DIM) prunes configs that would make
# the K^T tile wider than tall, which the chosen layout cannot handle.
#
# Backward pass uses a different scheme: two fixed sizes,
#   BLOCK_SIZE_MACRO = 128  -- the "outer" tile that stays resident
#   BLOCK_SIZE_MICRO = 32   -- the "inner" tile that streams past it
# The two backward kernels swap which axis gets which size:
#   _attn_bwd_dk_dv : BLOCK_KV = MACRO (resident),  BLOCK_Q = MICRO (streamed)
#                     -- one program owns a K/V tile, accumulating dK/dV by
#                        sweeping all Q blocks against it.
#   _attn_bwd_dq    : BLOCK_Q  = MACRO (resident),  BLOCK_KV = MICRO (streamed)
#                     -- one program owns a Q tile, accumulating dQ by
#                        sweeping all K/V blocks against it.
# This dual layout lets each kernel keep the bigger, more-reused tile in
# registers while paying only the smaller transfer cost per inner step.
# Both bwd kernels assume SEQ_LEN is divisible by BLOCK_SIZE_MACRO and
# BLOCK_SIZE_MICRO (they use plain // division for the launch grid and
# inner-loop trip count -- no tail handling).
# ---------------------------------------------------------------------------


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
    # Pick the [lo, hi) range of K/V positions this call is responsible for.
    # See the STAGE table at the top of the file for the meaning of each case.
    if STAGE == 1:
        # Causal: strictly to the left of the diagonal block -> no masking needed.
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Causal: the single diagonal block where unmasked/masked keys mix.
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        # Hint to the compiler that lo is BLOCK_SIZE_Q-aligned so it can
        # generate vectorized loads / skip bound checks.
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Non-causal: walk the whole K/V sequence in one pass.
        lo, hi = 0, SEQ_LEN

    # Move the K and V block pointers to the start of our [lo, hi) window.
    # K is laid out transposed (HEAD_DIM, SEQ_LEN), so we advance along axis 1.
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Online-softmax loop: stream K/V blocks, accumulate O, l_i, m_i in registers.
    # Invariant at the top of each iteration:
    #   m_i = running rowwise max of QK over keys seen so far
    #   l_i = running rowwise sum of exp(QK - m_i) over keys seen so far
    #   O_block = running weighted sum of V, rescaled to the current m_i / l_i
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Tell the compiler start_kv is BLOCK_SIZE_KV-aligned (enables vectorization).
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # ---- QK^T for this (Q block, K block) tile -----------------------
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            # Diagonal block of causal attention: apply the q >= k mask.
            # Adding -1e6 to masked entries pushes them to ~0 after exp().
            # Note that the scale is applied *before* the mask addition so the
            # mask sentinel doesn't get scaled.
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            # New running max (per query row).
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Unmasked path: fuse the scale into the max computation to save a mul.
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # P_ij = exp(QK_ij - m_ij). Stable because every entry is <= 0.
        P_block = tl.math.exp(QK_block)
        # Row sums of the (rescaled) attention probabilities for this tile.
        l_ij = tl.sum(P_block, 1)

        # Correction factor for the previously accumulated l_i / O_block,
        # to rebase them from the old running max (m_i) to the new one (m_ij).
        alpha = tl.math.exp(m_i - m_ij)
        # Rebase the denominator, then fold in this tile's contribution.
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # Rebase the partial output, then accumulate P @ V using tl.dot's
        # built-in accumulator argument so the FMA happens in one fused op.
        # O_new = alpha * O_old + P @ V
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        # Slide the running max forward.
        m_i = m_ij

        # Advance to the next K/V tile in the [lo, hi) window.
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


# Autotune sweep: 2 * 2 * 3 * 2 = 24 configs benchmarked once per
# (SEQ_LEN, HEAD_DIM) cache key, then cached. See the "Block sizes" section
# at the top of the file for what each axis controls.
#   BLOCK_SIZE_Q  in {64, 128}   -- query rows per program
#   BLOCK_SIZE_KV in {32, 64}    -- K/V rows per inner-loop step
#   num_stages    in {3, 4, 7}   -- software-pipelined load/compute stages
#   num_warps     in {2, 4}      -- warps per program (CTA width)
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
    # The chosen K^T tile shape is (HEAD_DIM, BLOCK_SIZE_KV), which only
    # tiles cleanly when BLOCK_SIZE_KV <= HEAD_DIM. This prunes invalid
    # autotune configs at compile time rather than failing at runtime.
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # Grid layout: axis 0 ranges over Q blocks within a sequence, axis 1
    # ranges over (batch * head) pairs. Each program handles one Q block of
    # one (batch, head) -- the corresponding row of the output O.
    block_index_q = tl.program_id(0)

    # Decompose the flattened (batch, head) program id back into its parts.
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # Byte offset into Q/K/V/O for this (batch, head). We promote to int64 to
    # avoid overflow on large tensors (BATCH * NUM_HEADS * SEQ_LEN * HEAD_DIM
    # can exceed 2^31 elements).
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

    # K is materialized as its transpose: shape (HEAD_DIM, SEQ_LEN). By
    # swapping the strides we get K^T "for free" so that the QK product can
    # be computed as Q @ K_T directly with a single tl.dot.
    #
    # `order` is a permutation of the axes ranking them from fastest-varying
    # in memory to slowest. The compiler uses it to emit coalesced loads,
    # pick the right shared-memory swizzle, and feed tensor cores efficiently.
    # The original K tensor is row-major (SEQ_LEN, HEAD_DIM), so HEAD_DIM has
    # stride 1 -- it is the contiguous axis. After the stride swap above,
    # HEAD_DIM is logical axis 0, hence order=(0, 1) ("axis 0 first").
    # Contrast with Q/O below, where HEAD_DIM stays as logical axis 1, so
    # they use order=(1, 0). Wrong `order` is still correct, just slower.
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),
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

    # Absolute query positions handled by this program.
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # Relative key/value positions inside one K/V tile.
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Online softmax state, one per query row in this block.
    #   m_i: running max, initialized to -inf (any finite QK will replace it).
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    #   l_i: running denominator. Starts at 1.0 (not 0.0) so the first
    #   correction factor exp(m_i_old - m_i_new) = exp(-inf - x) = 0
    #   cleanly zeroes it out before we add the first real l_ij.
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # Accumulator for the partial output O = softmax(QK^T) @ V.
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Load the Q block once; it stays resident in registers/SRAM for the
    # entire K/V streaming loop below (this is the core of FlashAttention).
    Q_block = tl.load(Q_block_ptr)

    # ---- Run one or two inner passes depending on causality --------------
    # STAGE == 1 (non-causal): single pass over all K/V (inner STAGE = 3).
    # STAGE == 3 (causal):     first pass over the unmasked left region
    #                          (inner STAGE = 1), then a second pass over the
    #                          diagonal block with masking (inner STAGE = 2).
    if STAGE == 1 or STAGE == 3:
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
        # Causal: second pass over the diagonal block, applying the q >= k mask.
        # ("Right of the diagonal" is fully masked out and never visited.)
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
    # ---- Epilogue --------------------------------------------------------
    # Save logsumexp = m + log(l) per query row. The backward pass uses this
    # single scalar to recover P = exp(QK*scale - logsumexp) without rerunning
    # the (numerically delicate) online softmax.
    m_i += tl.math.log(l_i)
    # Finalize O by dividing the accumulated numerator by the running sum.
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------
# The backward pass for attention needs the quantity
#   D_i = rowsum(O_i * dO_i)   (elementwise product, then sum over HEAD_DIM)
# which appears in the dS formula  dS = P * (dP - D).  We precompute D once
# per query row here so the dQ and dK/dV kernels can just load it.
# Like the forward pass, the backward uses a STAGE flag where STAGE == 3 means
# "causal -> apply the q >= k mask"; any other value means "no mask".
# ---------------------------------------------------------------------------


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)
    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    )
    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)
    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
    # Store the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Each program fixes one Q block and iterates over all K/V blocks, so we
    # parallelize across (Q block, batch*head) pairs.
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # M and D are indexed as (batch, head, seq), so we use a different offset.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Slide all the per-tensor base pointers to this (batch, head). We use raw
    # pointer arithmetic (not make_block_ptr) so we can build masks from the
    # absolute row/col indices when STAGE == 3.
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)

    # This program owns one BLOCK_Q-sized chunk of queries.
    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Load Q, dO, and the per-row logsumexp M (and D) for this Q block once;
    # they stay in registers while we stream over K/V below.
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    # K^T and V^T pointers: we read K and V as (HEAD_DIM, BLOCK_KV) blocks so
    # that QK^T = Q @ K^T and dP = dO @ V^T can be expressed as single tl.dot
    # calls with no explicit transpose.
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    # Walk every K/V block to accumulate the full dQ for our Q block.
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        # Recompute attention probabilities from the saved logsumexp M:
        #   P = exp(scale * QK^T - logsumexp).
        # This is the FlashAttention recomputation trick - we trade FLOPs for
        # memory by not materializing P during the forward pass.
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            # Causal: zero out probabilities for keys above the query (k > q).
            # No -inf trick needed since M already encodes the correct normalizer.
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        # Standard attention-backward identities:
        #   dP = dO @ V^T
        #   dS = P * (dP - D)   where D_i = sum_j O_ij * dO_ij
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        # dQ += scale * dS @ K. The scale is folded in here once per tile.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Advance to the next K/V tile.
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    # dK and dV accumulators for this K/V block, kept in registers/SRAM.
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # Load the K and V tile this program is responsible for once; they stay
    # resident while we stream over every Q block.
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # (BLOCK_KV, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # (BLOCK_KV, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # Q is read as Q^T: swap (offs_q, offs_dim) so each load yields a
    # (HEAD_DIM, BLOCK_Q) tile. Then K @ Q^T directly gives P^T, which is the
    # natural shape for accumulating dV = P^T @ dO and dK = dS^T @ Q.
    # dO is loaded in its normal (BLOCK_Q, HEAD_DIM) orientation.
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Stream over every Q block, accumulating into dK_block and dV_block.
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        # Logsumexp for the current Q block (one scalar per query row).
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        # P^T = exp(scale * K @ Q^T - logsumexp).
        # Computing P in transposed form avoids an explicit tl.trans later.
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Causal: mask_block is True where the (k, q) pair is allowed
            # (q >= k). Since m already absorbed the correct normalizer, we
            # only need to zero out disallowed entries -- no -inf needed.
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )  # Shape: (BLOCK_KV, BLOCK_Q)
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # dV += P^T @ dO  (accumulates contributions from this Q block).
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # D_i = rowsum(O_i * dO_i), precomputed in _attn_bwd_preprocess.
        Di = tl.load(D + offs_q)

        # dP^T = V @ dO^T  (transposed form, matching P^T's layout).
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # dS^T = P^T * (dP^T - D^T). The transpose of the standard
        # dS = P * (dP - D) identity from the FlashAttention paper.
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # dK += scale * dS^T @ Q. We multiply by Q (not Q^T) by transposing
        # the qT_block we already loaded.
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Advance to the next Q block.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Write back dV.
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK.
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # `ctx` is the per-call context object PyTorch's autograd machinery
        # passes to a custom torch.autograd.Function. It is the bridge between
        # forward() and backward(): anything stashed on `ctx` during forward
        # is visible to backward() when it runs later. We use it for two things:
        #   1) ctx.save_for_backward(...)  -> stores tensors that participate
        #      in autograd's graph; they reappear as ctx.saved_tensors in
        #      backward() with version counters checked to detect in-place
        #      mutation between the two passes.
        #   2) ctx.<attr> = value          -> stores plain Python state
        #      (scalars, flags, the launch grid) that we want available in
        #      backward but doesn't need autograd bookkeeping.
        # `ctx` is created fresh for each forward call, so there's no shared
        # state across invocations.
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        # All three projections must share a head dimension; the kernel assumes
        # a single HEAD_DIM constant for Q, K, and V tiles.
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # Output buffer matches Q exactly (same shape, dtype, device, strides).
        O = torch.empty_like(Q)
        # See the STAGE table at the top of the file: 1 = non-causal, 3 = causal.
        stage = 3 if causal else 1

        # Launch grid: one program per (Q block, batch*head). BLOCK_SIZE_Q is
        # chosen by the autotuner so we resolve the grid via a lambda.
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M stores the per-row logsumexp produced by the forward kernel; the
        # backward pass uses it to reconstruct softmax probabilities cheaply.
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # Launch the forward kernel. Triton accepts Python tensors directly
        # and passes their data pointers to the GPU; the strides are passed
        # explicitly so the kernel can index arbitrary memory layouts.
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

        # Stash everything backward() will need. Tensors that take part in the
        # autograd graph (Q, K, V, plus O and M which are derived from them)
        # go through save_for_backward so autograd can version-check them.
        ctx.save_for_backward(Q, K, V, O, M)
        # Plain Python attributes for non-tensor state. The grid lambda is
        # cached so backward could relaunch with the same tile shape if needed.
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        # Hand-tuned launch parameters (no autotune in the backward).
        NUM_WARPS, NUM_STAGES = 4, 3
        # Backward block sizes -- see the "Block sizes" section at the top
        # of the file. MICRO is the streamed tile (smaller, traversed in the
        # inner loop); MACRO is the resident tile (larger, owned by one
        # program). The two backward kernels swap which axis (Q or K/V) gets
        # which size:
        #   _attn_bwd_dk_dv  -> BLOCK_KV = MACRO, BLOCK_Q  = MICRO
        #   _attn_bwd_dq     -> BLOCK_Q  = MACRO, BLOCK_KV = MICRO
        # Both kernels divide SEQ_LEN by BLOCK_SIZE_MACRO with no remainder
        # handling, so SEQ_LEN must be a multiple of 128 for the bwd path.
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # Precompute D_i = rowsum(O_i * dO_i) once, reused by both bwd kernels.
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        # Pass 1: each program fixes one K/V block (macro) and streams over
        # every Q block (micro), accumulating dK and dV for that K/V block.
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # Pass 2: dual layout. Each program fixes one Q block (macro) and
        # streams over every K/V block (micro), accumulating dQ for that Q block.
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None


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
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("PASSED")
