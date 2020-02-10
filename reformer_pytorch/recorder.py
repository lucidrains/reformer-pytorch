from torch import nn
from reformer_pytorch.reformer_pytorch import LSHAttention, LSHSelfAttention
from collections import defaultdict

class Recorder(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.iter = 0
        self.recordings = defaultdict(list)
        self.net = net
        self.on = True

        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = True
            if isinstance(module, LSHSelfAttention):
                module.callback = self.record

    def eject(self):
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = False
            if isinstance(module, LSHSelfAttention):
                module.callback = None
        return self.net

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False

    def clear(self):
        del self.recordings
        self.recordings = defaultdict(list)

    def record(self, attn, buckets):
        if not self.on: return
        data = {'attn': attn.cpu(), 'buckets': buckets.cpu()}
        self.recordings[self.iter].append(data)

    def forward(self, x):
        out = self.net(x)
        self.iter += 1
        return out
