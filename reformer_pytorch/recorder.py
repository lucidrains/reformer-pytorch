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
        self.ejected = False

    def eject(self):
        self.ejected = True
        self.clear()
        self.unwire()
        return self.net

    def wire(self):
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = True
            if isinstance(module, LSHSelfAttention):
                module.callback = self.record

    def unwire(self):
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = False
            if isinstance(module, LSHSelfAttention):
                module.callback = None

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False

    def clear(self):
        del self.recordings
        self.recordings = defaultdict(list)
        self.iter = 0        

    def record(self, attn, buckets):
        if not self.on: return
        data = {'attn': attn.detach().cpu(), 'buckets': buckets.detach().cpu()}
        self.recordings[self.iter].append(data)

    def forward(self, x, **kwargs):
        assert not self.ejected, 'Recorder has already been ejected and disposed'
        if self.on:
            self.wire()

        out = self.net(x, **kwargs)

        self.iter += 1
        self.unwire()
        return out
