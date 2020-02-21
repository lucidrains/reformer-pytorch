import math
from torch import nn
import torch.nn.functional as F

from reformer_pytorch.reformer_pytorch import Reformer, ReformerLM, \
    LSHSelfAttention, LSHAttention


def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=0)


class Autopadder(nn.Module):
    def __init__(self, module):
        super().__init__()

        allowed_classes = (Reformer, ReformerLM, LSHAttention, LSHSelfAttention)
        assert isinstance(module, allowed_classes), \
            'Autopadder only accepts Reformer, ReformerLM, ' \
            'LSHAttention and LSHSelfAttention'
        self.module = module
        self.pad_dim = -1 if isinstance(module, ReformerLM) else -2

    def forward(self, x, **kwargs):
        b, t, m = *x.shape[:2], self.module.num_mem_kv

        keys = kwargs.get('keys')
        input_mask = kwargs.get('input_mask')
        input_attn_mask = kwargs.get('input_attn_mask')

        k_len = 0 if keys is None else keys.shape[1]
        seqlen = t + m + k_len

        if seqlen > self.module.full_attn_thres:
            multiple = self.module.bucket_size * 2
            x = pad_to_multiple(x, seqlen, multiple, dim=self.pad_dim)

            if input_mask is not None:
                diff = x.shape[1] - input_mask.shape[1]
                new_mask = F.pad(input_mask, (0, diff), value=False)
                kwargs.update(input_mask=new_mask)

            if input_attn_mask is not None:
                offset = x.shape[1] - input_attn_mask.shape[1]
                new_mask = F.pad(
                    input_attn_mask, (0, offset, 0, offset), value=False)
                kwargs.update(input_attn_mask=new_mask)

        out = self.module(x, **kwargs)
        return out[:, 0:t]
