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
    full_attn_thres = 1024, # use full attention if context length is less than set value
    use_scale_norm = False  # use scale norm from 'Transformers without tears' paper
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

## Examples

A full Reformer sequence → sequence, say translation

```python
import torch
from reformer_pytorch import ReformerLM

DE_SEQ_LEN = 4096
EN_SEQ_LEN = 4096

encoder = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    depth = 12,
    heads = 8,
    max_seq_len = DE_SEQ_LEN,
    fixed_position_emb = True,
    return_embeddings = True # return output of last attention layer
).cuda()

decoder = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    depth = 12,
    heads = 8,
    max_seq_len = EN_SEQ_LEN,
    fixed_position_emb = True,
    causal = True
).cuda()

x  = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
yi = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long().cuda()

enc_keys = encoder(x)               # (1, 4096, 1024)
yo = decoder(yi, keys = enc_keys)   # (1, 4096, 20000)
```

A full Reformer image → caption

```python
import torch
from torch.nn import Sequential
from torchvision import models
from reformer_pytorch import Reformer, ReformerLM

resnet = models.resnet50(pretrained=True)
resnet = Sequential(*list(resnet.children())[:-4])

SEQ_LEN = 4096

encoder = Reformer(
    dim = 512,
    depth = 6,
    heads = 8,
    max_seq_len = 4096,
)

decoder = ReformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    heads = 8,
    max_seq_len = SEQ_LEN,
    causal = True
)

x  = torch.randn(1, 3, 512, 512)
yi = torch.randint(0, 20000, (1, SEQ_LEN)).long()

visual_emb = resnet(x)
b, c, h, w = visual_emb.shape
visual_emb = visual_emb.view(1, c, h * w).transpose(1, 2) # nchw to nte

enc_keys = encoder(visual_emb)
yo = decoder(yi, keys = enc_keys) # (1, 4096, 20000)
```

## Benchmarks

- @zbloss has kindly added the GLUE benchmarks under the `glue/` directory.

## Todo

1. ~~Make it so Reformer can be used as decoder where queries only attend to fed key/values~~
2. ~~All-attention learned memory key values~~
3. ~~Option to switch to full shared-qk attention at shorter sequence lengths (< 2048 or a set threshold)~~
4. Recurrence like Transformer XL

## Citations
```bibtex
@inproceedings{kitaev2020reformer,
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

```bibtex
@article{1910.05895,
    author  = {Toan Q. Nguyen and Julian Salazar},
    title   = {Transformers without Tears: Improving the Normalization of Self-Attention},
    year    = {2019},
    eprint  = {arXiv:1910.05895},
    doi     = {10.5281/zenodo.3525484},
}
```

[♥](https://www.youtube.com/watch?v=GUo2XuqMcCU)