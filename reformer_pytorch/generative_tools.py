from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from reformer_pytorch.reformer_pytorch import ReformerLM
from reformer_pytorch.autopadder import Autopadder

class TrainingWrapper(nn.Module):
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        assert isinstance(net, ReformerLM), 'generative trainer wrapper can only accept ReformerLM class'
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = Autopadder(net)

    def forward(self, x, return_loss = False, **kwargs):
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        assert self.training, 'you must be training in order to return the loss'

        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        out = self.net(xi, **kwargs)

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss
