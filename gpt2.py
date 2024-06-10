from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class MLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') ## GELU is better for the gradients below zero, because it's not a flat line so there is always a reason to move. No dead neuron. Approximation is for historical reasons, because of computation limits.
        self.c_proj = nn.Linear(4* config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) ## First layer normalization -- normalized shape
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) ## We could have name it as feed forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) ## We first layer normalize and make the relations being noticed by the network via attention, reduce operation
        x = x + self.mlp(self.ln_2(x)) ## We then add second layer norm, and then the map operation
        ## Here is the interesting part we don't operate on residuals inside the gradients. Our residuals are clean and added by themselves.
        return x

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), ## Token embeddings -- num of embeddings, embedding dimension
            wpe = nn.Embedding(config.block_size, config.n.embd), ## Positional embeddings -- num of embeddings, embedding dimension
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ## Hidden layer of the transformer block
            ln_f = nn.LayerNorm(config.n_embd) ## Final Layer normalization -- it is different than original transformer
        ))
        self.lm_head = nn.Linear(config.n.embd, config.vocab_size, bias=False) ## Final linear layer-- input dimension, output dimension