from reformer_pytorch import ReformerLM

import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100

SEQ_LEN = 1024

# instantiate model

model = ReformerLM(
    emb = 512,
    depth = 6,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 64,
    n_hashes = 8,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = True,
    causal = True
)

model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq[0:-1].cuda(), full_seq[1:].cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_loader = iter(DataLoader(TextSamplerDataset(data_train, SEQ_LEN), batch_size = BATCH_SIZE))
val_loader = iter(DataLoader(TextSamplerDataset(data_val, SEQ_LEN), batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

def get_batch_loss(model, data):
    x, y = data
    pred = model(x)
    return F.cross_entropy(pred.transpose(1, 2), y, reduction='mean')

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = get_batch_loss(model, next(train_loader))
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i != 0 and i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = get_batch_loss(model, next(val_loader))
            print(f'validation loss: {loss.item()}')
