from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024 ## max sequence length
    vocab_size: int = 50527 ## number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token. "I'll cover tokenization a little bit as well!!!"
    n_layer: int = 12 ## num of layers
    n_head: int = 12 ## num of heads
    n_embd: int = 768 ## embedding dimension

class CasualSelfAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0 ## This is the general convention. The embedding is going to be build by the concatenation of the heads so it should be divisible by the n_head.
        ## key, query, value projections for all the heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        ## output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        ## it is not bias actually, it is the mask, lower triangular matrix but in this code we're following HF/OpenAI naming conventions.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) ## view is a reshape operation that reshapes our tensors.
        
    def forward(self, x):
        B, T, C = x.size() ## batch size, sequence length, embedding dimensionality (n_embd)
        ## nh is "number of heads", hs is "head size", C is "number of channels" = (nh * hs)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        ## Here is the attention calculations after getting q, k, v tensors
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) ## query * key transpoze divided by the hs of key.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) ## fill the parts coming after current sequence lenght with negative infinity.
        att = F.softmax(att, dim=-1) ## get the logits from the query and the key interactions and make them as probabilities.
        y = att @ v ## (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) ## re-assemble all head outputs side by side and reshape them.
        ## output projection
        y = self.c_proj(y)
        return y

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
            wpe = nn.Embedding(config.block_size, config.n_embd), ## Positional embeddings -- num of embeddings, embedding dimension
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ## Hidden layer of the transformer block
            ln_f = nn.LayerNorm(config.n_embd) ## Final Layer normalization -- it is different than original transformer
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) ## Final linear layer-- input dimension, output dimension

    def forward(self, idx):
        ## idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"You have to use {self.config.block_size} sequence length or less"
        ## forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) ## shape (T)
        pos_emb = self.transformer.wpe(pos) ## position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) ## token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        ## forward the block of transformer
        for block in self.transformer.h:
            x = block(x)
        ## final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) ## (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
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
    
num_of_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

import tiktoken
enc = tiktoken.get_encoding('gpt2') ## get the gpt2 tokenizer
tokens = enc.encode("Hello, I'm a language model,") ## tokenize the given sentence with gpt2 tokenizer algorithm
tokens = torch.tensor(tokens, dtype=torch.long) ## (8,) change this tokens to torch tensor
tokens = tokens.unsqueeze(0).repeat(num_of_return_sequences, 1) ## (5, 8) add another dimension and repeat the same values for num_of_return_sequences times.
## These tokens are the idx in our GPT class forward method. The parameter.
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    ## forward the model to get the logits
    with torch.no_grad():
        logits = model(x) ## (B, T, vocab_size)
        ## Take the logits' last position item
        logits = logits[:, -1, :] ## (B, vocab_size)
        ## Get the probabilities
        probs = F.softmax(logits, dim=-1)
        ## We'll do the top k sampling here (HF's default 50 for pipeline)
        ## (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ## Select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1) ## (B, 1)
        ## gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) ## (B, 1)
        ## append to the sequence to get the full generated sentences
        x = torch.cat((x, xcol), dim=1)

## decode and print the generated text
for i in range(num_of_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)



