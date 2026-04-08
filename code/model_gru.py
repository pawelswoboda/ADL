"""
Full definition of a GRU Language Model, all of it in this single file.
Designed to be comparable in parameter count (~124M) to GPT-2.

Default config: n_layer=10, n_embd=1080, vocab_size=50304 -> ~124.4M params
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GRUConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 10
    n_embd: int = 1080      # embedding dim and GRU hidden size (smaller than RNN due to 3x gate params)
    dropout: float = 0.0
    bias: bool = True
    custom_init: bool = True


class GRUCell(nn.Module):
    """
    GRU cell:
        r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)   (reset gate)
        z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)   (update gate)
        n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  (new gate)
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        # Batched linear projections for all 3 gates: reset, update, new
        self.i2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.hidden_size = hidden_size

    def forward(self, x, h):
        i_r, i_z, i_n = self.i2h(x).chunk(3, dim=-1)
        h_r, h_z, h_n = self.h2h(h).chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h
        return h_new


class GRU(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.cells = nn.ModuleList([
            GRUCell(config.n_embd, config.n_embd, bias=config.bias)
            for _ in range(config.n_layer)
        ])
        self.drop_rnn = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: embedding and output projection share the same weight matrix
        self.wte.weight = self.lm_head.weight

        # init all weights
        if config.custom_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, GRUCell):
            # orthogonal init for hidden-to-hidden weights helps gradient flow
            # apply to each gate's slice separately
            for i in range(3):
                start = i * module.hidden_size
                end = (i + 1) * module.hidden_size
                nn.init.orthogonal_(module.h2h.weight[start:end])
                nn.init.normal_(module.i2h.weight[start:end], mean=0.0, std=0.02)
            if module.i2h.bias is not None:
                nn.init.zeros_(module.i2h.bias)
                nn.init.zeros_(module.h2h.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        x = self.drop(self.wte(idx))
        hidden = [torch.zeros(B, self.config.n_embd, device=x.device, dtype=x.dtype)
                  for _ in range(self.config.n_layer)]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                hidden[l] = cell(inp, hidden[l])
                inp = self.drop_rnn(hidden[l]) if l < self.config.n_layer - 1 else hidden[l]
            outputs.append(inp)
        x = self.ln_f(torch.stack(outputs, dim=1))

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
        Uses GRU hidden state for efficient autoregressive generation.
        """
        B = idx.size(0)
        # process conditioning sequence to build up hidden state
        x = self.drop(self.wte(idx))
        hidden = [torch.zeros(B, self.config.n_embd, device=x.device, dtype=x.dtype)
                  for _ in range(self.config.n_layer)]
        for l, cell in enumerate(self.cells):
            h = hidden[l]
            outputs = []
            for t in range(x.size(1)):
                h = cell(x[:, t, :], h)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)
            hidden[l] = h
            if l < self.config.n_layer - 1:
                x = self.drop_rnn(x)
        logits = self.lm_head(self.ln_f(x[:, -1, :]))

        for _ in range(max_new_tokens):
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # forward single token through GRU, reusing hidden state
            x = self.drop(self.wte(idx_next))
            for l, cell in enumerate(self.cells):
                hidden[l] = cell(x[:, 0, :], hidden[l])
                x = hidden[l].unsqueeze(1)
            logits = self.lm_head(self.ln_f(x[:, 0, :]))

        return idx
