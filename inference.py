import torch
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig
import tiktoken



max_length = 200
model = GPT(GPTConfig(vocab_size=50304))
checkpoint = torch.load('log/model_04576.pt')
model_state_dictionary = checkpoint['model']

state_dict = {key.replace('_orig_mod.', ''): value for key, value in model_state_dictionary.items()}
text = "Hello I'm a language model, and"
B, T = 8, 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(state_dict)
model.to(device)
model.eval()
num_of_return_sequences = 5
enc = tiktoken.get_encoding('gpt2')

tokens = enc.encode(text)

#model.eval()

# Prepare input data (replace 'input_ids' with your actual input tensor)
input_ids = torch.tensor([tokens], dtype=torch.long)  # Shape: [1, sequence_length]

# Move the model and input data to the appropriate device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
last = None
while input_ids.size(1) < max_length:
    ## forward the model to get the logits
    with torch.no_grad():
        logits, _ = model(input_ids) ## (B, T, vocab_size)
        ## Take the logits' last position item
        logits = logits[:, -1, :] ## (B, vocab_size)
        ## Get the probabilities
        probs = F.softmax(logits, dim=-1)
        ## We'll do the top k sampling here (HF's default 50 for pipeline)
        ## (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        topk_indices = topk_indices
        ## Select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1) ## (B, 1)
        ## gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) ## (B, 1)
        ## append to the sequence to get the full generated sentences
        input_ids = torch.cat((input_ids, xcol), dim=1)


    

# Convert logits to probabilities and get the most likely next token ID
# probs = F.softmax(logits[:, -1, :], dim=-1)
# next_token_id = torch.argmax(probs, dim=-1)

# Print the predicted next token ID
print(input_ids)

## decode and print the generated text

tokens = input_ids[0, :max_length].detach().to('cpu').tolist()
# tokens = [int(token) for token in tokens if token != 0]
decoded = enc.decode(tokens)
print(">", decoded)