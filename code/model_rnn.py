"""
Full definition of a Vanilla RNN Language Model, all of it in this single file.
Designed to be comparable in parameter count (~124M) to GPT-2.

Default config: n_layer=10, n_embd=1536, vocab_size=50304 -> ~124.5M params

Usage:
    pytest model_rnn.py          # run correctness tests
    python model_rnn.py          # run micro-benchmark (needs CUDA)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class RNNConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 10
    n_embd: int = 1536      # embedding dim and RNN hidden size
    dropout: float = 0.0
    bias: bool = True
    custom_init: bool = True


class StackedRNNCells(nn.Module):
    """All RNN layers stored as stacked weight tensors for batched computation.

    h_t^l = tanh(W_ih^l @ x_t^l + bias_ih^l + W_hh^l @ h_{t-1}^l + bias_hh^l)

    Weights are stored as (n_layers, hidden_size, input_size) so that multiple
    layers can be computed in a single torch.bmm call.
    """

    def __init__(self, n_layers, input_size, hidden_size, bias=True):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.empty(n_layers, hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(n_layers, hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(n_layers, hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(n_layers, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward_sequential(self, x, drop_fn=None, hidden=None):
        """Process a sequence with the legacy column-by-column schedule.

        For each time step t, process all L layers top-to-bottom before
        moving to t+1.  Simple but sequential: depth = T * L.

            step 0: (t=0, l=0)
            step 1: (t=0, l=1)
            ...
            step L-1: (t=0, l=L-1)
            step L:   (t=1, l=0)    <- waits for ALL of t=0
            ...
            Total sequential steps: T * L

        Args:
            x:       (B, T, H) input embeddings
            drop_fn: dropout function applied between layers (not after the last)
            hidden:  optional list of L tensors (B, H); initialized to zeros if None
        Returns:
            output:  (B, T, H) final layer's output at every time step
            hidden:  list of L tensors (B, H) — final hidden state of each layer
        """
        B, T, _ = x.size()
        L = self.n_layers
        if hidden is None:
            hidden = [torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
                      for _ in range(L)]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l in range(L):
                b_ih = self.bias_ih[l] if self.bias_ih is not None else None
                b_hh = self.bias_hh[l] if self.bias_hh is not None else None
                hidden[l] = torch.tanh(F.linear(inp, self.W_ih[l], b_ih) + F.linear(hidden[l], self.W_hh[l], b_hh))
                inp = drop_fn(hidden[l]) if drop_fn and l < L - 1 else hidden[l]
            outputs.append(inp)
        return torch.stack(outputs, dim=1), hidden

    def forward_wavefront(self, x, drop_fn=None, hidden=None):
        """Process a sequence with wavefront (anti-diagonal) scheduling.

        Dependency structure of an L-layer RNN over T time steps
        (example: T=6, L=4):

            Layer
              3 | (0,3)-->(1,3)-->(2,3)-->(3,3)-->(4,3)-->(5,3)  <- output
                |   ^       ^       ^       ^       ^       ^
              2 | (0,2)-->(1,2)-->(2,2)-->(3,2)-->(4,2)-->(5,2)
                |   ^       ^       ^       ^       ^       ^
              1 | (0,1)-->(1,1)-->(2,1)-->(3,1)-->(4,1)-->(5,1)
                |   ^       ^       ^       ^       ^       ^
              0 | (0,0)-->(1,0)-->(2,0)-->(3,0)-->(4,0)-->(5,0)
                |   ^       ^       ^       ^       ^       ^
                |  x_0     x_1     x_2     x_3     x_4     x_5
                +------------------------------------------------> time

        Each cell (t, l) depends on exactly two predecessors:
          --> horizontal: hidden state from same layer at t-1  (recurrent)
           ^  vertical:   output of layer l-1 at same t        (depth)

        The legacy schedule processes column by column: complete all L
        layers for t=0 before starting t=1.  Only one cell runs at a time.
        Sequential depth = T * L.

        The wavefront schedule processes ANTI-DIAGONALS d = t + l.
        Cells on the same diagonal are fully independent: each cell's
        two predecessors lie on strictly earlier diagonals.

        Diagonal numbers across the grid (d = t + l):

            Layer
              3 |  3       4       5       6       7       8
              2 |  2       3       4       5       6       7
              1 |  1       2       3       4       5       6
              0 |  0       1       2       3       4       5
                +------------------------------------------------> time
                  t=0     t=1     t=2     t=3     t=4     t=5

        Execution schedule — K cells computed per diagonal:

            d=0: (t=0,l=0)                                K=1  ramp up
            d=1: (t=1,l=0), (t=0,l=1)                     K=2
            d=2: (t=2,l=0), (t=1,l=1), (t=0,l=2)          K=3
            d=3: (t=3,l=0), (t=2,l=1), (t=1,l=2), (t=0,l=3)  K=4  full width
            d=4: (t=4,l=0), (t=3,l=1), (t=2,l=2), (t=1,l=3)  K=4
            d=5: (t=5,l=0), (t=4,l=1), (t=3,l=2), (t=2,l=3)  K=4
            d=6:            (t=5,l=1), (t=4,l=2), (t=3,l=3)   K=3  ramp down
            d=7:                       (t=5,l=2), (t=4,l=3)   K=2
            d=8:                                  (t=5,l=3)    K=1

            Sequential depth = T + L - 1 = 9   (vs T*L = 24 for legacy)

        For T=1024, L=10:  1033 steps  vs  10240  — ~10x reduction.

        Why this maps well to the GPU:

        On each diagonal, the K independent cells use DIFFERENT weight
        matrices (one per layer). We cannot merge them into a single large
        matmul — but torch.bmm does exactly this: it takes K matrix pairs
        and runs K independent matmuls in ONE kernel launch.

        The weights W_ih, W_hh are stored as (L, H, H) tensors.  Slicing
        W_ih[l_min:l_max] returns a contiguous VIEW (no copy) of shape
        (K, H, H).  We stack the K input vectors and K hidden-state vectors
        into (K, B, H) tensors (a cheap gather of small vectors).  Then:

            torch.bmm( (K, B, H),  (K, H, H) )  -->  (K, B, H)

        This replaces K tiny matmuls launched in K separate CUDA kernels
        with one larger batched kernel, giving the GPU more work per launch
        and better SM utilization.

        Each diagonal follows a GATHER -> COMPUTE -> SCATTER pattern:
          GATHER:  collect inputs and hidden states for all active layers
          COMPUTE: one _forward_batched (two bmm calls + bias + tanh)
          SCATTER: write new hidden states back, apply inter-layer dropout

        Args:
            x:       (B, T, H) input embeddings
            drop_fn: dropout function applied between layers (not after the last)
            hidden:  optional list of L tensors (B, H); initialized to zeros if None
        Returns:
            output:  (B, T, H) final layer's output at every time step
            hidden:  list of L tensors (B, H) — final hidden state of each layer
        """
        B, T, _ = x.size()
        L = self.n_layers

        if hidden is None:
            hidden = [torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
                      for _ in range(L)]
        # inter[l][t] = output of layer l at time t
        # (post-dropout for l < L-1, raw for the final layer)
        inter = [[None] * T for _ in range(L)]

        # Sweep through T + L - 1 anti-diagonals
        for d in range(T + L - 1):
            # Active layers on this diagonal: l in [l_min, l_max).
            # t = d - l must satisfy 0 <= t < T  and  0 <= l < L.
            l_min = max(0, d - T + 1)
            l_max = min(d + 1, L)
            layers = slice(l_min, l_max)  # contiguous slice into weight tensors

            # --- GATHER ---
            # For each active layer l on this diagonal, collect:
            #   - input: embedding x[:,t,:] for layer 0, or inter[l-1][t] for deeper layers
            #   - hidden: the current hidden state of layer l (from the previous diagonal)
            inp_list = []
            h_list = []
            for l in range(l_min, l_max):
                t = d - l
                inp_list.append(x[:, t, :] if l == 0 else inter[l - 1][t])
                h_list.append(hidden[l])
            inputs = torch.stack(inp_list)   # (K, B, H) — K small vectors stacked
            hiddens = torch.stack(h_list)    # (K, B, H)

            # --- COMPUTE ---
            # One batched matmul replaces K sequential matmuls.
            # W_ih[layers] is a contiguous view (zero-copy) of shape (K, H, H).
            #   new_h = tanh( bmm(inputs, W_ih.T) + bmm(hiddens, W_hh.T) + bias )
            W_ih = self.W_ih[layers]  # (K, H, H) — view, no copy
            W_hh = self.W_hh[layers]  # (K, H, H) — view, no copy
            pre = torch.bmm(inputs, W_ih.transpose(1, 2)) + torch.bmm(hiddens, W_hh.transpose(1, 2))
            if self.bias_ih is not None:
                pre = pre + (self.bias_ih[layers] + self.bias_hh[layers]).unsqueeze(1)
            new_h = torch.tanh(pre)  # (K, B, H)

            # --- SCATTER ---
            # Write each layer's new hidden state back and store the inter-layer
            # activation.  Dropout is applied between layers (not after the last).
            for i, l in enumerate(range(l_min, l_max)):
                t = d - l
                hidden[l] = new_h[i]
                inter[l][t] = drop_fn(new_h[i]) if drop_fn and l < L - 1 else new_h[i]

        # Collect the final layer's output at every time step -> (B, T, H)
        return torch.stack([inter[L - 1][t] for t in range(T)], dim=1), hidden


class VanillaRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.cells = StackedRNNCells(config.n_layer, config.n_embd, config.n_embd, bias=config.bias)
        self.drop_rnn = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: embedding and output projection share the same weight matrix
        self.wte.weight = self.lm_head.weight

        # init all weights
        if config.custom_init:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, StackedRNNCells):
            # orthogonal init for hidden-to-hidden weights helps gradient flow
            for l in range(module.n_layers):
                nn.init.orthogonal_(module.W_hh[l])
            nn.init.normal_(module.W_ih, mean=0.0, std=0.02)
            if module.bias_ih is not None:
                nn.init.zeros_(module.bias_ih)
                nn.init.zeros_(module.bias_hh)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x, _ = self.cells.forward_wavefront(self.drop(self.wte(idx)), drop_fn=self.drop_rnn)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def forward_legacy(self, idx, targets=None):
        x, _ = self.cells.forward_sequential(self.drop(self.wte(idx)), drop_fn=self.drop_rnn)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # Note: stacked biases (bias_ih, bias_hh) are 2D but should not be decayed.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'bias' not in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'bias' in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
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
        # rough estimate: ~6N flops per token (2N forward, 4N backward)
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Uses RNN hidden state for efficient autoregressive generation.
        """
        # Process the conditioning sequence to build up hidden state
        x = self.drop(self.wte(idx))
        output, hidden = self.cells.forward_sequential(x, drop_fn=self.drop_rnn)
        logits = self.lm_head(self.ln_f(output[:, -1, :]))

        for _ in range(max_new_tokens):
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Forward single token through RNN, reusing hidden state
            x = self.drop(self.wte(idx_next))
            output, hidden = self.cells.forward_sequential(x, drop_fn=self.drop_rnn, hidden=hidden)
            logits = self.lm_head(self.ln_f(output[:, 0, :]))

        return idx


# ---------------------------------------------------------------------------
# Tests — run with: pytest model_rnn.py -v
# ---------------------------------------------------------------------------

def _make_model(bias=True, n_layer=4, n_embd=16):
    config = RNNConfig(
        block_size=32, vocab_size=64, n_layer=n_layer, n_embd=n_embd,
        dropout=0.0, bias=bias, custom_init=True,
    )
    model = VanillaRNN(config)
    model.eval()
    return model


@torch.no_grad()
def test_logits_match_with_targets():
    model = _make_model()
    torch.manual_seed(42)
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    logits_wf, loss_wf = model.forward(idx, targets)
    logits_lg, loss_lg = model.forward_legacy(idx, targets)
    assert torch.allclose(logits_wf, logits_lg, atol=1e-5)
    assert abs(loss_wf.item() - loss_lg.item()) < 1e-5


@torch.no_grad()
def test_logits_match_without_targets():
    model = _make_model()
    torch.manual_seed(7)
    idx = torch.randint(0, 64, (3, 24))
    logits_wf, loss_wf = model.forward(idx)
    logits_lg, loss_lg = model.forward_legacy(idx)
    assert loss_wf is None and loss_lg is None
    assert torch.allclose(logits_wf, logits_lg, atol=1e-5)


@torch.no_grad()
def test_logits_match_no_bias():
    model = _make_model(bias=False)
    torch.manual_seed(99)
    idx = torch.randint(0, 64, (2, 20))
    targets = torch.randint(0, 64, (2, 20))
    logits_wf, loss_wf = model.forward(idx, targets)
    logits_lg, loss_lg = model.forward_legacy(idx, targets)
    assert torch.allclose(logits_wf, logits_lg, atol=1e-5)
    assert abs(loss_wf.item() - loss_lg.item()) < 1e-5


def test_gradients_match():
    model = _make_model()
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    _, loss_wf = model.forward(idx, targets)
    loss_wf.backward()
    grads_wf = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    _, loss_lg = model.forward_legacy(idx, targets)
    loss_lg.backward()

    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(grads_wf[n], p.grad, atol=1e-5), \
                f"grad mismatch in {n}: max diff {(grads_wf[n] - p.grad).abs().max().item()}"


# ---------------------------------------------------------------------------
# Benchmark — run with: python model_rnn.py
#
# Uses torch.utils.benchmark which handles CUDA synchronization, warmup,
# and statistical reporting (median, IQR) automatically.
# Sized for a 16 GB GPU (RTX 4070 Ti Super): 124M-param model, B=4, T=256.
# ---------------------------------------------------------------------------

def benchmark():
    import torch.utils.benchmark as bench

    assert torch.cuda.is_available(), "Benchmark requires CUDA"
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = RNNConfig()  # default 124M params
    model = VanillaRNN(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B, T = 4, 256
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)
    print(f"Input: B={B}, T={T}\n")

    results = []

    # ---- forward only ----
    for sub_label, method in [('sequential', 'forward_legacy'), ('wavefront', 'forward')]:
        fn = getattr(model, method)
        t = bench.Timer(
            stmt='fn(idx)',
            globals={'fn': fn, 'idx': idx},
            label='forward',
            sub_label=sub_label,
            description='124M RNN',
        )
        results.append(t.blocked_autorange(min_run_time=2))

    # ---- forward + backward ----
    for sub_label, method in [('sequential', 'forward_legacy'), ('wavefront', 'forward')]:
        fn = getattr(model, method)
        t = bench.Timer(
            stmt='_, loss = fn(idx, targets); loss.backward(); model.zero_grad(set_to_none=True)',
            globals={'fn': fn, 'idx': idx, 'targets': targets, 'model': model},
            label='forward+backward',
            sub_label=sub_label,
            description='124M RNN',
        )
        results.append(t.blocked_autorange(min_run_time=2))

    bench.Compare(results).print()


if __name__ == "__main__":
    benchmark()
