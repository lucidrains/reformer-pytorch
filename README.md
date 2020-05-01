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
    ff_dropout = 0.1,
    post_attn_dropout = 0.1,
    layer_dropout = 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    emb_dim = 128,        # embedding factorization for further memory savings
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    attn_chunks = 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    twin_attention = False, # both branches of the reversible network will be attention
    full_attn_thres = 1024, # use full attention if context length is less than set value
    reverse_thres = 1024,   # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
    use_scale_norm = False,  # use scale norm from 'Transformers without tears' paper
    use_rezero = False,      # remove normalization and use rezero from 'ReZero is All You Need'
    one_value_head = False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
    weight_tie = False,           # tie parameters of each layer for no memory per additional depth
    weight_tie_embedding = False, # use token embedding for projection of output, some papers report better results
    n_local_attn_heads = 2,       # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
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

## Masking

This repository supports masks on the input sequence `input_mask (b x i_seq)`, the context sequence `context_mask (b x c_seq)`, as well as the rarely used full attention matrix itself `input_attn_mask (b x i_seq x i_seq)`, all made compatible with LSH attention. Masks are made of booleans where `False` denotes masking out prior to the softmax.

The causal triangular mask is all taken care of for you if you set `causal = True`.

```python
import torch
from reformer_pytorch import ReformerLM

CONTEXT_LEN = 512
SEQ_LEN = 8192

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 1,
    max_seq_len = SEQ_LEN,
    ff_chunks = 8,
    causal = True
)

c = torch.randn(1, CONTEXT_LEN, 1024)
x = torch.randint(0, 20000, (1, SEQ_LEN)).long()

i_mask = torch.ones(1, SEQ_LEN).bool()
c_mask = torch.ones(1, CONTEXT_LEN).bool()

y = model(x, keys = c, input_mask = i_mask, context_mask = c_mask)
# masking done correctly in LSH attention
```

## Positional Embeddings

<a href="https://github.com/AranKomat">Aran</a> has informed me that the Reformer team used axial position embeddings with great results on longer sequences. I tested it out and indeed it works very well! So well in fact that I have decided to make this the default. You can adjust the shape and dimension of the axial embeddings by following the instructions below.


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
    axial_position_shape = (128, 64),  # the shape must multiply up to the max_seq_len (128 x 64 = 8192)
    axial_position_dims = (512, 512)   # the dims must sum up to the model dimensions (512 + 512 = 1024)
)

x = torch.randint(0, 20000, (1, 8192)).long()
y = model(x) # (1, 8192, 20000)
```

If you would rather use absolute positional embeddings, you can turn it on with `absolute_position_emb = True` flag on initialization.

## Training

Since version `0.17.0`, and some corrections to the reversible network, Reformer Pytorch is compatible with Microsoft's Deepspeed! If you have multiple local GPUs, you can follow the instructions / example <a href="https://github.com/lucidrains/reformer-pytorch/tree/master/examples/enwik8_deepspeed">here</a>.

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
    max_seq_len = 4096
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

## Reformer Encoder Decoder Architecture

**There is a bug in versions < `0.21.0`. Please upgrade to at least the version specified for the working encoder / decoder Reformer.**

By popular demand, I have coded up a wrapper that removes a lot of the manual work in writing up a generic Reformer encoder / decoder architecture. To use, you would import the `ReformerEncDec` class. Encoder keyword arguments would be passed with a `enc_` prefix and decoder keyword arguments with `dec_`. The model dimension (`dim`) must be prefix free and will be shared between encoder and decoder. The framework will also take care of passing the encoder input mask to the decoder context mask, unless explicitly overridden.

```python
import torch
from reformer_pytorch import ReformerEncDec

DE_SEQ_LEN = 4096
EN_SEQ_LEN = 4096

enc_dec = ReformerEncDec(
    dim = 512,
    enc_num_tokens = 20000,
    enc_depth = 6,
    enc_max_seq_len = DE_SEQ_LEN,
    dec_num_tokens = 20000,
    dec_depth = 6,
    dec_max_seq_len = EN_SEQ_LEN
).cuda()

train_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
train_seq_out = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long().cuda()
input_mask = torch.ones(1, DE_SEQ_LEN).bool().cuda()

loss = enc_dec(train_seq_in, train_seq_out, return_loss = True, enc_input_mask = input_mask)
loss.backward()
# learn

# evaluate with the following
eval_seq_in = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
eval_seq_out_start = torch.tensor([[0.]]).long().cuda() # assume 0 is id of start token
samples = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len = EN_SEQ_LEN, eos_token = 1) # assume 1 is id of stop token
print(samples.shape) # (1, <= 1024) decode the tokens
```

## Customizing Feedforward

By default, the activation function is `GELU`. If you would like an alternative activation function, you can pass in the class to the keyword `ff_activation`.

```python
import torch
from reformer_pytorch import ReformerLM
from torch import nn

model = ReformerLM(
    num_tokens= 20000,
    dim = 512,
    depth = 6,
    max_seq_len = 8192,
    ff_chunks = 8,
    ff_dropout = 0.1,
    ff_mult = 6,
    ff_activation = nn.LeakyReLU,
    ff_glu = True # use GLU in feedforward, from paper 'GLU Variants Improve Transformer'
)

x = torch.randint(0, 20000, (1, 8192)).long()
y = model(x) # (1, 8192, 20000)
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
    title     = {Reducing Transformer Depth on Demand with Structured Dropout},
    author    = {Angela Fan and Edouard Grave and Armand Joulin},
    booktitle = {International Conference on Learning Representations},
    year      = {2020},
    url       = {https://openreview.net/forum?id=SylO2yStDr}
}
```

```bibtex
@article{Shazeer2019FastTD,
    title   = {Fast Transformer Decoding: One Write-Head is All You Need},
    author  = {Noam Shazeer},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1911.02150}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}    
}
```

```bibtex
@misc{roy*2020efficient,
    title   = {Efficient Content-Based Sparse Attention with Routing Transformers},
    author  = {Aurko Roy* and Mohammad Taghi Saffar* and David Grangier and Ashish Vaswani},
    year    = {2020},
    url     = {https://openreview.net/forum?id=B1gjs6EtDr}
}
```

```bibtex
@misc{bachlechner2020rezero,
    title   = {ReZero is All You Need: Fast Convergence at Large Depth},
    author  = {Thomas Bachlechner and Bodhisattwa Prasad Majumder and Huanru Henry Mao and Garrison W. Cottrell and Julian McAuley},
    year    = {2020},
    url     = {https://arxiv.org/abs/2003.04887}
}
```

[♥](https://www.youtube.com/watch?v=GUo2XuqMcCU)
