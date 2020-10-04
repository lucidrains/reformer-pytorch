import math
import torch
from torch import nn
import torch.nn.functional as F

from reformer_pytorch.reformer_pytorch import Reformer, ReformerLM, LSHSelfAttention

def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=0)

class Autopadder(nn.Module):
    def __init__(self, net):
        super().__init__()
        assert isinstance(net, (LSHSelfAttention, Reformer, ReformerLM)), 'only modules LSHSelfAttention, Reformer, ReformerLM accepted'
        self.net = net

        reformer = net.reformer if isinstance(net, ReformerLM) else net
        self.pad_dim = -1 if isinstance(net, ReformerLM) else -2

        self.bucket_size = reformer.bucket_size
        self.num_mem_kv = reformer.num_mem_kv
        self.full_attn_thres = reformer.full_attn_thres

    def forward(self, x, **kwargs):
        b, t, m, device = *x.shape[:2], self.num_mem_kv, x.device

        keys = kwargs.get('keys')
        input_mask = kwargs.get('input_mask')
        input_attn_mask = kwargs.get('input_attn_mask')

        k_len = 0 if keys is None else keys.shape[1]
        seqlen = t + m + k_len

        if seqlen > self.full_attn_thres:
            if input_mask is None:
                input_mask = torch.full((b, t), True, device=x.device, dtype=torch.bool)

            x = pad_to_multiple(x, seqlen, self.bucket_size * 2, dim=self.pad_dim)

            if input_mask is not None:
                new_mask = F.pad(input_mask, (0, x.shape[1] - input_mask.shape[1]), value=False)
                kwargs.update(input_mask=new_mask)

            if input_attn_mask is not None:
                offset = x.shape[1] - input_attn_mask.shape[1]
                new_mask = F.pad(input_attn_mask, (0, offset, 0, offset), value=False)
                kwargs.update(input_attn_mask=new_mask)

        out = self.net(x, **kwargs)
        return out[:, 0:t]
