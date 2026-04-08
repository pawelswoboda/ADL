# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
# ---

# %% [markdown]
## GPT2: Efficient Implementation, Decoder Architecture, Training
#- We will implement [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), OpenAI's early breakthrough Transformer decoder. 
#- We will investigate a few improvements/modifications used by GPT-2.
#- We will write reasonably efficient code.
#- Based on [Karpathy's lecture](https://www.youtube.com/watch?v=l8pRSuU81PU).

# %%
# 1: Imports
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

# %% [markdown]
### Configuration: dataclasses
#- We will use dataclasses for holding configuration parameters.
#- Good practice, done by many projects

# %%
# 2: configuration
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

config = GPTConfig()

# %% [markdown]
### Efficient Attention
#- Multihead attention in one module
#- One linear layer for all Q,K,V-matrices and split afterwards
#- One linear layer for output projection
#- Multihead attention through reshaping and adding an extra dimension

# %%
# 3: Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self ,x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        #att = (q @ k.transpose(-2, -1)) 
        att = torch.einsum("BQhd,BKhd->BhQK", [q, k])
        att /= math.sqrt(self.n_embd // self.n_head)

        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        #y = att @ v # (B, nh, T, T) x (B, T, nh, hs) -> (B, T, nh, hs)
        out = torch.einsum("BhQK,BKhd->BQhd", [att, v])
        out = out.reshape(B, T, self.n_embd) # -> (B, T, C)
        out = self.c_proj(out)

        return out

# %%
# Test CausalSelfAttention

if __name__ == "__main__":
    x = torch.randn(4, 512, config.n_embd)
    start_time = time.time()
    csa = CausalSelfAttention(config)
    execution_time = time.time() - start_time
    print(f"Execution time attention: {execution_time} seconds")
    print(csa(x).shape)

# %% [markdown]
### Faster Attention
#- Right now we explicitly compute and store the attention matrix.
# - This can be memory intensive, since the attention matrix is of size (B, T, T) and scales quadratically with sequence length.
#  - Interestingly, computing the attention matrix is not the bottleneck, but storing it to memory for the backward pass
#- Pytorch uses the [FlashAttention](https://arxiv.org/abs/2205.14135) algorithm to speed up the attention computation and circumvent storing the attention matrix.

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

# %%
if __name__ == "__main__":
    x = torch.randn(4, 512, config.n_embd)
    start_time = time.time()
    csa = CausalSelfAttention(config)
    execution_time = time.time() - start_time
    print(f"Execution time flash attention: {execution_time} seconds")
    print(csa(x).shape)

# %%
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
# %%

# %% [markdown]
### Residual Connections
#We can write the forward pass of each decoder layer of the original Transformer architecture as
#
#```
#x = LayerNorm1(x + Attn(x))
#x = LayerNorm2(x + FFN(x))
#```
#
#This is suboptimal, since there is no uninterrupted residual path from input to output. 
#In particular, the residuals take part in the normalization. Instead, for example in GPT-2, the normalization is moved. The resulting decoder layer's forward pass becomes
#
#```
#x = x + Attn(LayerNorm1(x))
#x = x + FFN(LayerNorm2(x))
#```
#
#Now there exists a direct, clean, unmodified, residual stream from input to output.
#
# We will put everything together in a decoder layer.

# %%
# GPT decoder layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerDecoderLayer, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# %% [markdown]
### Learnable positional encoding
#- Instead of non-learned sinusiodal waves for encoding positions, we instead learn positional encoding.
#- Instead of proper GELU, we use approximate GELU, for historical reasons, see [here](https://github.com/pytorch/pytorch/issues/39853).


# %% [markdown]
### Shared Embedding and Output Head
#
#- Note that the embedding is a matrix of size vocab_size $\times$ dim, whereas the head is of size dim $\times$ vocab_size.
#- Since often we have a very large vocab_size (~50.000 for GPT-2) and the transformer dim is also large ($\in [768,1600]$), the embedding and the output layer actually constitute a large fraction of the parameters.
#- For efficiency, the output head is often taken as the transpose of the embedding, see [here](https://arxiv.org/abs/1608.05859). This has also been done in the original transformer paper and the GPT-2 implementation.
#
#We also slightly rewrite the class so that the OpenAI implementation will match variable by variable.

# %%
class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.transformer.wpe.weight.shape[0], "Cannot forward sequence longer than block_size."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# %%
# Test GPT
if __name__ == "__main__":
    vocab_size = 256
    x = torch.randint(0, vocab_size, (4, 8)).to(device)
    gpt = GPT(config).to(device)
    y = gpt(x)
    print(y.shape)

# %% [markdown]
## GPT-2
#- We have everything ready to get the [GPT-2 model](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). 
#- There are multiple versions of GPT-2 with the same architecture but with different number of layers, attention heads and dimensions.
#- For economical reasons we will take the smallest variant with 124M parameters.
#- We will download the weights and load them into our Transformer decoder.
#- We will also use the tokenizer used by GPT-2.

# %%
# Read in pretrained weights
def GPT2_from_pretrained():
    """Loads pretrained GPT-2 model weights from huggingface"""
    model_type = 'gpt2'
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # create a from-scratch initialized minGPT model
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model

if __name__ == "__main__":
    model = GPT2_from_pretrained()
    model.to(device)

# %% [markdown]
### Tokenizer and decoding
#- We use the tiktoken tokenizer with GPT-2 settings.
#- We autoregressively decode text:
#    - start with a prompt
#    - sample a token from the model
#    - append the token to the prompt
#    - repeat until max length
#- We use top-k sampling with k=50, which is the default for huggingface pipelines.
#- We use the bfloat16 datatype for faster inference on GPUs with tensor cores (NVIDIA A100, V100, T4, etc.) and MPS (Apple silicon).

# %%
import tiktoken
if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    device_type = 'cuda' if device == torch.device('cuda') else 'mps' if device == torch.device('mps') else 'cpu'

def generate(model, prompt, max_length=32):
    model.eval()
    num_return_sequences = 4
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # use bfloat16 for faster inference
                logits = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # decode the generated indices
    decoded = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded.append(enc.decode(tokens))
    
    return decoded

# %%
# Test decoding and time it

if __name__ == "__main__":
    start_time = time.time()
    text = generate(model, "I am Donald Trump and I will", max_length=512)
    end_time = time.time()
    for i, t in enumerate(text):
        print(f"Sample {i}: {t}\n")

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

