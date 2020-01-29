## Reformer, the Efficient Transformer, in Pytorch
[![PyPI version](https://badge.fury.io/py/reformer-pytorch.svg)](https://badge.fury.io/py/reformer-pytorch)

<img src="./lsh_attention.png" width="500">

This is a Pytorch implementation of Reformer https://openreview.net/pdf?id=rkgNKkHtvB

It includes LSH attention, reversible network, and chunking. It has been validated with an auto-regressive task (enwik8). It also includes additional features to make the entire network pure attention all the way down.

Test 32k tokens with Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1am1DRl80Kd3o6n_4u3MomPzYS0NfdHAC)

## Install

```bash
$ pip install reformer_pytorch
```

## Usage

A simple Reformer language model

```python
# should fit in ~ 5gb - 8k tokens

import torch
from reformer_pytorch import ReformerLM

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    emb_dim = 128,        # embedding factorization for further memory savings
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    twin_attention = False, # both branches of the reversible network will be attention
    use_full_attn = False,  # use full self attention, for comparison
    full_attn_thres = 1024  # use full attention if context length is less than set value
).cuda()

x = torch.randint(0, 20000, (1, 8192)).long().cuda()
y = model(x) # (1, 8192, 20000)
```

The Reformer (just a stack of reversible LSH attention)

```python
# should fit in ~ 5gb - 8k embeddings

import torch
from reformer_pytorch import Reformer

model = Reformer(
    dim = 512,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
).cuda()

x = torch.randn(1, 8192, 512).cuda()
y = model(x) # (1, 8192, 512)
```

Self Attention with LSH

```python
import torch
from reformer_pytorch import LSHSelfAttention

attn = LSHSelfAttention(
    dim = 128,
    heads = 8,
    bucket_size = 64,
    n_hashes = 8,
    causal = False
)

x = torch.randn(10, 1024, 128)
y = attn(x) # (10, 1024, 128)
```

LSH (locality sensitive hashing) Attention

```python
import torch
from reformer_pytorch import LSHAttention

attn = LSHAttention(
    bucket_size = 64,
    n_hashes = 16,
    causal = True
)

qk = torch.randn(10, 1024, 128)
v = torch.randn(10, 1024, 128)

attn_out, buckets = attn(qk, v) # (10, 1024, 128)
# buckets will contain the bucket number (post-argmax) of each token of each batch
```

A full Reformer encoder / decoder architecture example

```python
import torch
from reformer_pytorch import Reformer, ReformerLM

DATA_LEN = 8192
SEQ_LEN = 4096

encoder = Reformer(
    dim = 512,
    depth = 12,
    heads = 8,
    max_seq_len = DATA_LEN
)

decoder = ReformerLM(
    num_tokens = 20000,
    dim = 512,
    emb_dim = 128,
    depth = 12,
    heads = 8,
    max_seq_len = SEQ_LEN,
    fixed_position_emb = True,
    causal = True
)

x = torch.randn(1, DATA_LEN, 512)
y = torch.randint(0, 20000, (1, SEQ_LEN)).long()

enc_keys = encoder(x)
o = decoder(y, keys = enc_keys) # (1, 4096, 20000)
```

## Todo

1. ~~Make it so Reformer can be used as decoder where queries only attend to fed key/values~~
2. ~~All-attention learned memory key values~~
3. Recurrence like Transformer XL
4. ~~Option to switch to full shared-qk attention at shorter sequence lengths (< 2048 or a set threshold)~~

## Citations
```bibtex
@inproceedings{
    kitaev2020reformer,
    title={Reformer: The Efficient Transformer},
    author={Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgNKkHtvB}
}
```

```bibtex
@article{DBLP:journals/corr/abs-1907-01470,
  author    = {Sainbayar Sukhbaatar and
               Edouard Grave and
               Guillaume Lample and
               Herv{\'{e}} J{\'{e}}gou and
               Armand Joulin},
  title     = {Augmenting Self-attention with Persistent Memory},
  journal   = {CoRR},
  volume    = {abs/1907.01470},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.01470}
}
```
