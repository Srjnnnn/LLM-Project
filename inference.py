import torch
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig
import tiktoken



max_length = 50
model = GPT(GPTConfig(vocab_size=50304))
checkpoint = torch.load('log/model_04576.pt')
model_state_dictionary = checkpoint['model']

state_dict = {key.replace('_orig_mod.', ''): value for key, value in model_state_dictionary.items()}
text = "Hello I'm a language model, and"
B, T = 8, 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(state_dict)

num_of_return_sequences = 1
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)

# Prepare input data (replace 'input_ids' with your actual input tensor)
input_ids = torch.tensor([tokens], dtype=torch.long)  # Shape: [1, sequence_length]

# Move the model and input data to the appropriate device (CPU or CUDA)
model.to(device)
input_ids = input_ids.to(device)

while input_ids.size(1) < max_length:
    ## forward the model to get the logits
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids) ## (B, T, vocab_size)
        ## Take the logits' last position item
        logits = logits[:, -1, :] ## (B, vocab_size)
        ## Get the probabilities
        probs = F.softmax(logits, dim=-1)
        ## We'll do the top k sampling here (HF's default 50 for pipeline)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1, largest=True) ## If you want some fun change the 1 to some other number and you'll have different answers!!!
        ## Select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1, generator=torch.cuda.manual_seed(42), replacement=True) ## (B, 1)
        ## gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) ## (B, 1)
        ## append to the sequence to get the full generated sentences
        input_ids = torch.cat((input_ids, xcol), dim=1)

# Print the predicted token IDs
print(input_ids)

## decode and print the generated text
tokens = input_ids[0, :max_length].detach().to('cpu').tolist()
decoded = enc.decode(tokens)
print(">", decoded)