# %% [markdown]
## Sequence Models
#- We have learned about one sequence model - Transformers
#  - Powerful, versatile
#  - But quadratic complexity due to attention
#- In the following we will learn about a number of other sequence models that have different representational/computational trade-offs.
#  - Recurrent Neural Networks (RNNs)
#  - Long Short-Term Memory (LSTMs)
#  - State Space Models (SSMs)
#    - Mamba1/2

# %% [markdown]
#### Fixed-Size Features
#- One can view the Transformer as having a linearly increasing presentation size with attention allowing information to flow between different token features
#  - Token mixing is performed between all tokens in parallel
#- In contract, RNNs, LSTMs and SSMs have fixed size features that represent all tokens until the current point in the sequence.
#  - Feature of a hidden state is accumulated recursively

# %% [markdown]
## RNNs
#- Image ack Stanford CS 230
#- The simplest sequence model
#
#<img src="./SequenceModels/rnn.png" width="600"/>
#
#- In detail, the update equations are
#
#$$
#\boxed{
#a^{\langle t \rangle} = g_1\left(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a\right)
#}
#$$
#
#and
#
#$$
#\boxed{
#y^{\langle t \rangle} = g_2\left(W_{ya} a^{\langle t \rangle} + b_y\right)
#}
#$$
#
#  - The parameters $W_{aa}$, $W_{ax}$, $W_{ya}$, $b_a$, $g_2$ and $b_y$ are shared temporally
#
#<img src="./SequenceModels/rnn_detail.png" width="600"/>

# %%
# Implementation Inference
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.W_ih = nn.Linear(hidden_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        x = self.embed(x)  # (B, T, H)
        B, T, _ = x.size()
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)
        outputs = []
        for t in range(T):
            h = torch.tanh(self.W_ih(x[:, t, :]) + self.W_hh(h))
            out = self.W_out(h)
            outputs.append(out.unsqueeze(1))  # (B, 1, V)
        return torch.cat(outputs, dim=1), h  # (B, T, V), (B, H)

# %%
# Generation
# Parameters
vocab_size = 65 # number of characters
hidden_size = 100
batch_size = 1

# Instantiate and evaluate
model = RNN(vocab_size, hidden_size, vocab_size)
model.eval()

# Dummy input
x0 = torch.randint(0, vocab_size, (batch_size, 1))

# Inference
with torch.no_grad():
    output, hidden = model(x0)

print("Output: 1", output)

# second generation
x1 = torch.randint(0, vocab_size, (batch_size, 1))
with torch.no_grad():
    output, hidden = model(x1, hidden)
print("Output 2:", output)

x = torch.cat([x0,x1], dim=1)
with torch.no_grad():
    output, hidden = model(x)

print("Output 2, equivalent computation:", output)


# %%
# Implementation Training

# Dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Load and preprocess
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s): return [stoi[c] for c in s]
def decode(t): return ''.join([itos[i] for i in t])
data = torch.tensor(encode(text), dtype=torch.long)

# Generate text 
def generate(model, start_str, length, device):
    model.eval()
    input_ids = torch.tensor([stoi[c] for c in start_str], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, h = model(input_ids)  # let RNN handle the full prompt
        idx = input_ids[:, -1].unsqueeze(0)  # start generation from last token
        generated = start_str
        for _ in range(length):
            logits, h = model(idx, h)
            prob = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.multinomial(prob, num_samples=1)
            idx.unsqueeze(0)
            generated += itos[idx.item()]
    return generated

# Example usage:
print(generate(model, start_str="ROMEO:", length=200, device='cpu'))


# Dataset
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y



# %%
# Training function
def train_model(model, device, epochs=50):
    block_size = 128
    batch_size = 512
    train_loader = DataLoader(CharDataset(data, block_size), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir="runs/rnn_train")

    model.to(device)
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        total_grad_sq = 0.0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            grad_sq = sum((p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None))
            grad_norm = grad_sq ** 0.5
            total_grad_sq += grad_sq
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar("Loss/Batch", loss.item(), global_step)
            writer.add_scalar("GradNorm/Batch", grad_norm, global_step)
            global_step += 1
        print(f"Epoch {epoch+1}, avg loss: {total_loss / len(train_loader):.4f}")
        sample = generate(model, start_str="ROMEO:", length=200, device=device)
        print(f"Epoch {epoch+1}, generated text: {sample}")
        writer.add_text("Generated/Text", sample, global_step)

    writer.close()

#if __name__ == "__main__"
#    model = RNN(vocab_size, hidden_size, vocab_size).to(device)
#    train_model(model, device, epochs=10)

# %% [markdown]
#### Deep RNNs
#- Naive: Model until now only has a single layer.
#- Better: Stack multiple RNN blocks on top of each other.
#  - More expressive and powerful model, similar to stacking multiple Transformer blocks
#
#<img src="./SequenceModels/deep_rnn.png" width="600"/>

# %%
import torch
import torch.nn as nn

class DeepRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Stack of RNN layers
        self.rnn_layers = nn.ModuleList([
            nn.ModuleDict({
                "W_ih": nn.Linear(hidden_size, hidden_size),
                "W_hh": nn.Linear(hidden_size, hidden_size)
            }) for _ in range(num_layers)
        ])
        
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        x = self.embed(x)  # (B, T, H)
        B, T, _ = x.size()
        
        # Initialize hidden state for each layer
        if h is None:
            h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for l, layer in enumerate(self.rnn_layers):
                h[l] = torch.tanh(layer["W_ih"](inp) + layer["W_hh"](h[l]))
                inp = h[l]  # input to next layer
            out = self.W_out(h[-1])  # output from topmost layer
            outputs.append(out.unsqueeze(1))  # (B, 1, V)
        
        return torch.cat(outputs, dim=1), h  # (B, T, V), list of (B, H)

# %%
# Train DeepRNN
if __name__ == "__main__":
    num_layers=8
    model = DeepRNN(vocab_size, hidden_size, vocab_size, num_layers).to(device)
    train_model(model, device)

# %% [markdown]
#- Advantages	
#  - Possibility of processing input of any length
#  - Model size not increasing with size of input
#  - Computation takes into account historical information
#  - Weights are shared across time
#- Drawbacks
#  - Computation slow during training
#    - No parallel pass as in attention
#  - Difficulty of accessing information from a long time ago
#    - Hidden State must remember all previous informatino
#  - Cannot out of the box consider any future input for the current state
#    - No mechanism like causal mask in attention that can be undone
#    - But bidirectional RNNs exist
#  - Hard to train:
#    - Gradient vanishes/explodes

# %% [markdown]
#### Applications
#- One-to-One, $T_x = T_y = 1$
#
#<img src="./SequenceModels/rnn_one_to_one.png" width="600"/>
#
#- One-to-Many, $T_x = 1 < T_y$
#
#<img src="./SequenceModels/rnn_one_to_many.png" width="600"/>
#
#- Many-to-One, $T_x > 1 = T_y$
#
#<img src="./SequenceModels/rnn_many_to_one.png" width="600"/>
#
#- Many-to-Many, $T_x = T_y$
#
#<img src="./SequenceModels/rnn_many_to_many_1.png" width="600"/>
#
#- Many-to-Many, $T_x \neq T_y$
#
#<img src="./SequenceModels/rnn_many_to_many_2.png" width="600"/>
#
