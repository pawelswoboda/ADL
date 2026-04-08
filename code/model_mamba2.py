"""
Full definition of a Mamba2 Language Model, all of it in this single file.
Based on "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality" (Dao & Gu, 2024).

Default config: n_layer=23, d_model=768, d_state=128, headdim=64 -> ~125M params

Usage:
    pytest model_mamba2.py          # run correctness tests
    python model_mamba2.py          # run micro-benchmark (needs CUDA)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@dataclass
class Mamba2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 23
    d_model: int = 768
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 64
    dropout: float = 0.0
    custom_init: bool = True


class Mamba2Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        d_inner = config.d_model * config.expand
        nheads = d_inner // config.headdim
        self.d_inner = d_inner
        self.nheads = nheads
        self.headdim = config.headdim
        self.d_state = config.d_state
        self.ngroups = config.ngroups
        self.chunk_size = config.chunk_size

        # input projection: x, z, B, C, dt
        d_proj = 2 * d_inner + 2 * config.ngroups * config.d_state + nheads
        self.in_proj = nn.Linear(config.d_model, d_proj, bias=False)

        # short causal depthwise convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=config.d_conv,
            groups=d_inner, padding=config.d_conv - 1,
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(nheads))
        # initialize dt_bias so softplus gives values in [0.001, 0.1]
        dt = torch.exp(torch.rand(nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))

        self.norm = RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.size()

        # project and split into x, z (gate), B, C, dt
        proj = self.in_proj(x)
        z, x_conv, B_ssm, C_ssm, dt = proj.split(
            [self.d_inner, self.d_inner,
             self.ngroups * self.d_state, self.ngroups * self.d_state,
             self.nheads], dim=-1)

        # causal conv1d + silu
        x_conv = F.silu(self.conv1d(x_conv.transpose(1, 2))[:, :, :T].transpose(1, 2))

        # reshape for multi-head SSM
        x_ssm = x_conv.view(B, T, self.nheads, self.headdim)
        B_ssm = B_ssm.view(B, T, self.ngroups, self.d_state)
        C_ssm = C_ssm.view(B, T, self.ngroups, self.d_state)

        # expand groups to heads
        if self.ngroups < self.nheads:
            heads_per_group = self.nheads // self.ngroups
            B_ssm = B_ssm.repeat_interleave(heads_per_group, dim=2)
            C_ssm = C_ssm.repeat_interleave(heads_per_group, dim=2)

        # discretize
        A = -torch.exp(self.A_log.float())  # (nheads,)
        dt = F.softplus(dt + self.dt_bias)  # (B, T, nheads)

        # chunk-based SSD (structured state space duality)
        CS = self.chunk_size
        n_chunks = (T + CS - 1) // CS
        T_pad = n_chunks * CS
        if T_pad > T:
            pad = T_pad - T
            x_ssm = F.pad(x_ssm, (0, 0, 0, 0, 0, pad))
            B_ssm = F.pad(B_ssm, (0, 0, 0, 0, 0, pad))
            C_ssm = F.pad(C_ssm, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))

        # reshape into chunks: (batch, n_chunks, chunk_size, ...)
        x_c = x_ssm.view(B, n_chunks, CS, self.nheads, self.headdim)
        B_c = B_ssm.view(B, n_chunks, CS, self.nheads, self.d_state)
        C_c = C_ssm.view(B, n_chunks, CS, self.nheads, self.d_state)
        dt_c = dt.view(B, n_chunks, CS, self.nheads)

        # cumulative log-decay within each chunk
        log_dA = dt_c * A  # (B, n_chunks, CS, nheads)
        log_dA_cum = torch.cumsum(log_dA, dim=2)  # (B, n_chunks, CS, nheads)

        # --- intra-chunk via matrix multiply ---
        # L[i,j] = decay from step j+1 to step i (causal)
        L = torch.exp(log_dA_cum.unsqueeze(3) - log_dA_cum.unsqueeze(2))  # (B, n_chunks, CS, CS, nheads)
        causal_mask = torch.tril(torch.ones(CS, CS, device=x.device, dtype=x.dtype))
        L = L * causal_mask[None, None, :, :, None]

        # CB[i,j] = C[i] . B[j] over d_state dimension
        CB = torch.einsum('bnihd,bnjhd->bnijh', C_c, B_c)

        # x scaled by dt
        x_dt = x_c * dt_c.unsqueeze(-1)  # (B, n_chunks, CS, nheads, headdim)

        # intra-chunk output
        y_intra = torch.einsum('bnijh,bnjhd->bnihd', L * CB, x_dt)

        # --- inter-chunk: propagate states across chunk boundaries ---
        # decay from each position to end of its chunk
        decay_to_end = torch.exp(log_dA_cum[:, :, -1:, :] - log_dA_cum)  # (B, n_chunks, CS, nheads)

        # chunk end state: h_end[n] = sum_t decay_to_end[t] * (x_dt[t] outer B[t])
        chunk_states = torch.einsum('bnth,bnthd,bnths->bnhds',
                                     decay_to_end, x_dt, B_c)  # (B, n_chunks, nheads, headdim, d_state)

        # total decay across each chunk
        chunk_decay = torch.exp(log_dA_cum[:, :, -1, :])  # (B, n_chunks, nheads)

        # sequential scan over chunks (n_chunks steps instead of T steps)
        states = torch.zeros(B, self.nheads, self.headdim, self.d_state,
                            device=x.device, dtype=x.dtype)
        all_states = []
        for i in range(n_chunks):
            all_states.append(states)
            states = chunk_decay[:, i, :, None, None] * states + chunk_states[:, i]
        initial_states = torch.stack(all_states, dim=1)  # (B, n_chunks, nheads, headdim, d_state)

        # decay from start of chunk to each position within chunk
        decay_from_start = torch.exp(log_dA_cum)  # (B, n_chunks, CS, nheads)

        # inter-chunk contribution: y_inter[t] = C[t] . (decay[t] * h_initial)
        y_inter = torch.einsum('bnth,bnhds,bnths->bnthd',
                                decay_from_start, initial_states, C_c)

        # combine intra + inter + skip connection
        y = y_intra + y_inter + self.D[None, None, None, :, None] * x_c

        # reshape back to (B, T, d_inner)
        y = y.reshape(B, T_pad, self.d_inner)[:, :T, :]
        return self.out_proj(self.norm(y) * F.silu(z))

    def forward_sequential(self, x):
        """Legacy sequential forward pass: step-by-step SSM recurrence."""
        B, T, _ = x.size()

        # project and split (same as efficient version)
        proj = self.in_proj(x)
        z, x_conv, B_ssm, C_ssm, dt = proj.split(
            [self.d_inner, self.d_inner,
             self.ngroups * self.d_state, self.ngroups * self.d_state,
             self.nheads], dim=-1)

        # causal conv1d + silu
        x_conv = F.silu(self.conv1d(x_conv.transpose(1, 2))[:, :, :T].transpose(1, 2))

        # reshape for multi-head SSM
        x_ssm = x_conv.view(B, T, self.nheads, self.headdim)
        B_ssm = B_ssm.view(B, T, self.ngroups, self.d_state)
        C_ssm = C_ssm.view(B, T, self.ngroups, self.d_state)

        # expand groups to heads
        if self.ngroups < self.nheads:
            heads_per_group = self.nheads // self.ngroups
            B_ssm = B_ssm.repeat_interleave(heads_per_group, dim=2)
            C_ssm = C_ssm.repeat_interleave(heads_per_group, dim=2)

        # discretize
        A = -torch.exp(self.A_log.float())  # (nheads,)
        dt = F.softplus(dt + self.dt_bias)  # (B, T, nheads)

        # sequential SSM recurrence
        h = torch.zeros(B, self.nheads, self.headdim, self.d_state,
                        device=x.device, dtype=x.dtype)
        y = torch.zeros(B, T, self.nheads, self.headdim,
                        device=x.device, dtype=x.dtype)

        for t in range(T):
            dt_t = dt[:, t, :]  # (B, nheads)
            dA = torch.exp(A * dt_t)  # (B, nheads)
            x_t = x_ssm[:, t, :, :]  # (B, nheads, headdim)
            x_dt_t = x_t * dt_t[:, :, None]  # (B, nheads, headdim)
            B_t = B_ssm[:, t, :, :]  # (B, nheads, d_state)
            C_t = C_ssm[:, t, :, :]  # (B, nheads, d_state)

            # h = dA * h + outer(x_dt, B)
            h = dA[:, :, None, None] * h + x_dt_t[:, :, :, None] * B_t[:, :, None, :]
            # y = C . h + D * x
            y[:, t] = torch.einsum('bhds,bhs->bhd', h, C_t) + self.D[None, :, None] * x_t

        y = y.reshape(B, T, self.d_inner)
        return self.out_proj(self.norm(y) * F.silu(z))


class Mamba2Layer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba = Mamba2Block(config)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Mamba2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Mamba2Layer(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        if config.custom_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.drop(self.wte(idx))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        T = cfg.block_size
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate by re-running the full sequence each time (simple, not optimized).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Benchmark — run with: python model_mamba2.py
#
# Uses torch.utils.benchmark which handles CUDA synchronization, warmup,
# and statistical reporting (median, IQR) automatically.
# ---------------------------------------------------------------------------

def _swap_to_sequential(model):
    """Replace all block forwards with sequential version, return originals."""
    orig = []
    for layer in model.layers:
        orig.append(layer.mamba.forward)
        layer.mamba.forward = layer.mamba.forward_sequential
    return orig


def _restore_forwards(model, orig):
    """Restore original block forwards."""
    for layer, fwd in zip(model.layers, orig):
        layer.mamba.forward = fwd


def benchmark():
    import torch.utils.benchmark as bench

    assert torch.cuda.is_available(), "Benchmark requires CUDA"
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    config = Mamba2Config()  # default ~125M params
    model = Mamba2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B, T = 4, 256
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)
    print(f"Input: B={B}, T={T}\n")

    results = []

    # ---- forward only ----
    for sub_label, use_sequential in [('sequential', True), ('chunked', False)]:
        if use_sequential:
            orig = _swap_to_sequential(model)
        t = bench.Timer(
            stmt='model(idx)',
            globals={'model': model, 'idx': idx},
            label='forward',
            sub_label=sub_label,
            description='125M Mamba2',
        )
        results.append(t.blocked_autorange(min_run_time=2))
        if use_sequential:
            _restore_forwards(model, orig)

    # ---- forward + backward ----
    for sub_label, use_sequential in [('sequential', True), ('chunked', False)]:
        if use_sequential:
            orig = _swap_to_sequential(model)
        t = bench.Timer(
            stmt='_, loss = model(idx, targets); loss.backward(); model.zero_grad(set_to_none=True)',
            globals={'model': model, 'idx': idx, 'targets': targets},
            label='forward+backward',
            sub_label=sub_label,
            description='125M Mamba2',
        )
        results.append(t.blocked_autorange(min_run_time=2))
        if use_sequential:
            _restore_forwards(model, orig)

    bench.Compare(results).print()


if __name__ == "__main__":
    benchmark()


# ---------------------------------------------------------------------------
# Tests — run with: pytest model_mamba2.py -v
# ---------------------------------------------------------------------------


def _make_model(n_layer=2, d_model=32, d_state=16, headdim=16, chunk_size=8):
    config = Mamba2Config(
        block_size=64, vocab_size=64, n_layer=n_layer, d_model=d_model,
        d_state=d_state, d_conv=4, expand=2, headdim=headdim, ngroups=1,
        chunk_size=chunk_size, dropout=0.0, custom_init=True,
    )
    model = Mamba2(config)
    model.eval()
    return model


def _run_both_block_forwards(block, x):
    """Run efficient and sequential forward on a single Mamba2Block."""
    y_eff = block.forward(x)
    y_seq = block.forward_sequential(x)
    return y_eff, y_seq


@torch.no_grad()
def test_block_outputs_match():
    """Efficient chunked and sequential forward produce the same output."""
    model = _make_model()
    block = model.layers[0].mamba
    torch.manual_seed(42)
    x = torch.randn(2, 16, model.config.d_model)
    y_eff, y_seq = _run_both_block_forwards(block, x)
    assert torch.allclose(y_eff, y_seq, atol=1e-4), \
        f"max diff {(y_eff - y_seq).abs().max().item()}"


@torch.no_grad()
def test_block_outputs_match_non_chunk_aligned():
    """Sequence length not divisible by chunk_size."""
    model = _make_model(chunk_size=8)
    block = model.layers[0].mamba
    torch.manual_seed(7)
    x = torch.randn(2, 13, model.config.d_model)  # 13 not divisible by 8
    y_eff, y_seq = _run_both_block_forwards(block, x)
    assert torch.allclose(y_eff, y_seq, atol=1e-4), \
        f"max diff {(y_eff - y_seq).abs().max().item()}"


@torch.no_grad()
def test_full_model_logits_match_with_targets():
    """Full model forward vs forward with sequential blocks, with targets."""
    model = _make_model()
    torch.manual_seed(42)
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    logits_eff, loss_eff = model(idx, targets)

    orig = _swap_to_sequential(model)
    logits_seq, loss_seq = model(idx, targets)
    _restore_forwards(model, orig)

    assert torch.allclose(logits_eff, logits_seq, atol=1e-4), \
        f"logit max diff {(logits_eff - logits_seq).abs().max().item()}"
    assert abs(loss_eff.item() - loss_seq.item()) < 1e-4


@torch.no_grad()
def test_full_model_logits_match_without_targets():
    """Full model forward without targets (inference mode)."""
    model = _make_model()
    torch.manual_seed(99)
    idx = torch.randint(0, 64, (3, 24))

    logits_eff, loss_eff = model(idx)

    orig = _swap_to_sequential(model)
    logits_seq, loss_seq = model(idx)
    _restore_forwards(model, orig)

    assert loss_eff is None and loss_seq is None
    assert torch.allclose(logits_eff, logits_seq, atol=1e-4), \
        f"logit max diff {(logits_eff - logits_seq).abs().max().item()}"


def test_gradients_match():
    """Gradients through efficient and sequential forward are consistent."""
    model = _make_model()
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    _, loss_eff = model(idx, targets)
    loss_eff.backward()
    grads_eff = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    orig = _swap_to_sequential(model)
    _, loss_seq = model(idx, targets)
    loss_seq.backward()
    _restore_forwards(model, orig)

    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(grads_eff[n], p.grad, atol=1e-4), \
                f"grad mismatch in {n}: max diff {(grads_eff[n] - p.grad).abs().max().item()}"