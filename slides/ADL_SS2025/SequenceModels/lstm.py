# %% [markdown]
## LSTM: Long Short-Term Memory
# Image ack Chris Olah
#- One of the main drawbacks of RNNs are their training problems:
#  - During backward pass multiplicative gradient updates can be exponentially decreasing/increasing with respect to the number of layers/sequence length
#- Remedies:
#  - Exploding gradient: Gradient clipping
#  - Vanishing gradient: 
#    - Gates for selectively adding/removing information to/from cell state
#    - An `information highway'' (similar to residual connection) for letting information flow through unrolled LSTM

# %% [markdown]
#### LSTM Architecture
# LSTM cell:
#
#<img src="./SequenceModels/lstm_module.png" width="400"/>
#
#  - $C$ = Upper cell state
#  - $h$ = Lower cell state
#  - $\times$ signifies elementwise multiplication
#- Gates:
#  - General Form: $\Gamma = \sigma\left(W x^{\langle t \rangle} + U a^{\langle t-1 \rangle} + b\right)$
#
#- Forget gate: Erase cell feature?
#
#<img src="./SequenceModels/lstm_forget_gate.png" width="400"/>
#
#- Input gate: Which values will we update? $i_t$.
#  - Candidate values added to cell state: $\tilde{C}_t$.
#
#<img src="./SequenceModels/lstm_update_gate1.png" width="400"/>
#
#- Update old cell state $C_{t-1}$ to new cell state $C_t$.
#
#<img src="./SequenceModels/lstm_update_gate2.png" width="400"/>
#
#- Output gate $\Gamma_o$: What to output given cell state $C_t$?
#
#<img src="./SequenceModels/lstm_output_gate.png" width="400"/>
#
#- Peephole connections: Let all gates access the cell state
#
#<img src="./SequenceModels/lstm_peephole_connection.png" width="400"/>
#

# %%
import torch
import torch.nn as nn

# LSTM implementation with multiple layers
class DeepLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm_layers = nn.ModuleList([
            nn.ModuleDict({
                "W_ih": nn.Linear(hidden_size, 4 * hidden_size),
                "W_hh": nn.Linear(hidden_size, 4 * hidden_size)
            }) for _ in range(num_layers)
        ])
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        x = self.embed(x)  # (B, T, H)
        B, T, _ = x.size()
        if state is None:
            h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = state

        outputs = []
        for t in range(T):
            input_t = x[:, t, :]
            for l in range(self.num_layers):
                gates = self.lstm_layers[l]["W_ih"](input_t) + self.lstm_layers[l]["W_hh"](h[l])
                i, f, g, o = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c[l] = f * c[l] + i * g
                h[l] = o * torch.tanh(c[l])
                input_t = h[l]
            out = self.W_out(h[-1])
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h, c)

# %%
# Train DeepLSTM
from rnn import train_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 65 # number of characters
hidden_size = 128
num_layers = 8

if __name__ == "__main__":
    model = DeepLSTM(vocab_size, hidden_size, vocab_size, num_layers).to(device)
    train_model(model, device)