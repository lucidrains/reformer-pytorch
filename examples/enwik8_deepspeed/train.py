import deepspeed

from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

import argparse
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

# constants

EPOCHS = 20
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 4096

# helpers

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

model = ReformerLM(
    dim = 512,
    depth = 6,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 64,
    n_hashes = 4,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = True,
    causal = True,
    n_local_attn_heads = 4,
    use_full_attn = False # set this to true for comparison with full attention
)

model = TrainingWrapper(model)
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
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

# setup deepspeed

cmd_args = add_argument()
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(),  training_data=train_dataset)

# training

for _ in range(EPOCHS):
    for i, data in enumerate(trainloader):
        model_engine.train()
        data = data.to(model_engine.local_rank)
        loss = model_engine(data, return_loss = True)
        model_engine.backward(loss)
        model_engine.step()
        print(loss.item() * GRADIENT_ACCUMULATE_EVERY)

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                inp = random.choice(val_dataset)[:-1]
                loss = model(inp[None, :].cuda(), return_loss = True)
                print(f'validation loss: {loss.item()}')

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp.cuda(), GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print(output_str)
