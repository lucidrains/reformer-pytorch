# Reformer, the Efficient Transformer, in Pytorch

This is a Pytorch implementation of Reformer https://openreview.net/pdf?id=rkgNKkHtvB

It includes LSH attention, reversible network, and chunking. It has been validated with a toy auto-regressive task.

## Prerequisites

```
torch
revtorch
```

Run in virtual env
```
> pip install -r requirements.txt
```

## Usage

The full Reformer

```
import torch
from reformer import Reformer

model = Reformer(
    emb = 512,
    depth = 12,
    max_seq_len = 1024,
    num_tokens= 20000,
    heads = 8,
    causal = True,       # auto-regressive or not
    bucket_size = 64,    # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 8,        # should keep at 8 per paper
    ff_chunks = 200,     # number of chunks for feedforward layer
    weight_tie = False   # tie parameters of each layer for no memory per additional depth
)

x = torch.randint(0, 20000, (1, 1024)).long()
y = model(x)
```

LSH Attention

```
import torch
from reformer import LSHSelfAttention

attn = LSHSelfAttention(
    emb = 128,
    heads = 8,
    bucket_size = 64,
    n_hashes = 8,
    causal = False
)

x = torch.randn(10, 1024, 128)
y = attn(x)
```

## Todo

1. Make it so Reformer can be used as decoder where queries only attend to fed key/values
2. Recurrence like Transformer XL
3. All-attention learned memory key values
