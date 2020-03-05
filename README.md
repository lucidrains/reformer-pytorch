## Reformer, the Efficient Transformer, in Pytorch
[![PyPI version](https://badge.fury.io/py/reformer-pytorch.svg)](https://badge.fury.io/py/reformer-pytorch)

<img src="./lsh_attention.png" width="500">

This is a Pytorch implementation of Reformer https://openreview.net/pdf?id=rkgNKkHtvB

It includes LSH attention, reversible network, and chunking. It has been validated with an auto-regressive task (enwik8). It also includes additional features to make the entire network pure attention all the way down.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1am1DRl80Kd3o6n_4u3MomPzYS0NfdHAC) 32k tokens

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1awNgXYtjvUeXl1gS-v1iyDXTJJ-fyJIK) 81k tokens with half precision

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
    layer_dropout = 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    emb_dim = 128,        # embedding factorization for further memory savings
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    attn_chunks = 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    twin_attention = False, # both branches of the reversible network will be attention
    full_attn_thres = 1024, # use full attention if context length is less than set value
    reverse_thres = 1024,   # turn off reversibility for 2x speed for sequence lengths shorter than designated value
    use_scale_norm = False,  # use scale norm from 'Transformers without tears' paper
    one_value_head = False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
    use_full_attn = False    # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
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

out, attn, buckets = attn(qk, v) # (10, 1024, 128)
# attn contains the unsorted attention weights, provided return_attn is set to True (costly otherwise)
# buckets will contain the bucket number (post-argmax) of each token of each batch
```

## Positional Embeddings

<a href="https://github.com/AranKomat">Aran</a> has informed me that the Reformer team used axial position embeddings with great results on longer sequences. I tested it out and indeed it works very well! If you choose to use it, you will have to pass in 2 additional hyperparameters in addition to turning on a flag.

It is highly recommended that you turn this on, especially if you are working with images. The Reformer team used an axial shape that matches the image dimensions of Imagenet `(64, 64, 3)`.

```python
import torch
from reformer_pytorch import ReformerLM

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 8192,
    ff_chunks = 8,
    attn_chunks = 2,
    causal = True,
    axial_position_emb = True,
    axial_position_shape = (128, 64),  # the shape must multiply up to the max_seq_len (128 x 64 = 8192)
    axial_position_dims = (512, 512)   # the dims must sum up to the model dimensions (512 + 512 = 1024)
)

x = torch.randint(0, 20000, (1, 8192)).long()
y = model(x) # (1, 8192, 20000)
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
    axial_position_emb = True,
    axial_position_shape = (32, 32),
    axial_position_dims = (256, 256)
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

## Research

To access the attention weights and bucket distribution, simply wrap the instantiated model with the `Recorder` wrapper class.

```python
import torch
from reformer_pytorch import Reformer, Recorder

model = Reformer(
    dim = 512,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
).cuda()

model = Recorder(model)

x = torch.randn(1, 8192, 512).cuda()
y = model(x)

model.recordings[0] # a list of attention weights and buckets for the first forward pass

model.turn_off() # stop recording
model.turn_on() # start recording
model.clear() # clear the recordings

model = model.eject() # recover the original model and remove all listeners
```

## Additional Helpers

Reformer comes with a slight drawback that the sequence must be neatly divisible by the bucket size * 2. I have provided a small helper tool that can help you auto-round the sequence length to the next best multiple.

```python
import torch
from reformer_pytorch import ReformerLM, Autopadder

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True,
    bucket_size = 63,   # odd bucket size
    num_mem_kv = 77     # odd memory key length
).cuda()

model = Autopadder(model)

SEQ_LEN = 7777 # odd sequence length
keys = torch.randn(1, 137, 1024) # odd keys length

x = torch.randint(0, 20000, (1, SEQ_LEN)).long().cuda()
y = model(x, keys = keys) # (1, 7777, 20000)
```

## Helpers for training auto-regressive models

A lot of users are only interested in an auto-regressive language model (like GPT-2). Here is a training wrapper to make it easy to both train and evaluate on arbitrarily lengthed sequences of encoded tokens. You will have to take care of the encoding and decoding yourself.

```python
import torch
from torch import randint

from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 4096,
    lsh_dropout = 0.1,
    causal = True,
    full_attn_thres = 1024
)

# 0 is used for padding and no loss to be calculated on it
model = TrainingWrapper(model, ignore_index = 0, pad_value = 0)

# the wrapper can handle evenly packed sequences
x_train = randint(0, 20000, (3, 357))

# or if you have a list of uneven sequences, it will be padded for you
x_train = [
    randint(0, 20000, (120,)),
    randint(0, 20000, (253,)),
    randint(0, 20000, (846,))
]

# when training, set return_loss equal to True
model.train()
loss = model(x_train, return_loss = True)
loss.backward()

# when evaluating, just use the generate function, which will default to top_k sampling with temperature of 1.
initial = torch.tensor([[0]]).long() # assume 0 is start token
sample = model.generate(initial, 100, temperature=1., filter_thres = 0.9, eos_token = 1) # assume end token is 1, or omit and it will sample up to 100
print(sample.shape) # (1, <=100) token ids
```

## Todo

1. ~~Make it so Reformer can be used as decoder where queries only attend to fed key/values~~
2. ~~All-attention learned memory key values~~
3. ~~Option to switch to full shared-qk attention at shorter sequence lengths (< 2048 or a set threshold)~~
4. Recurrence like Transformer XL

## Citations
```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
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

```bibtex
@inproceedings{fan2020reducing,
    title     ={Reducing Transformer Depth on Demand with Structured Dropout},
    author    ={Angela Fan and Edouard Grave and Armand Joulin},
    booktitle ={International Conference on Learning Representations},
    year      ={2020},
    url       ={https://openreview.net/forum?id=SylO2yStDr}
}
```

```bibtex
@article{Shazeer2019FastTD,
    title   ={Fast Transformer Decoding: One Write-Head is All You Need},
    author  ={Noam Shazeer},
    journal ={ArXiv},
    year    ={2019},
    volume  ={abs/1911.02150}
}
```

[♥](https://www.youtube.com/watch?v=GUo2XuqMcCU)