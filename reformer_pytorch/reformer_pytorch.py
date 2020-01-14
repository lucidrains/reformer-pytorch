import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from revtorch import ReversibleBlock, ReversibleSequence

# helper fns

def make_unit_length(x, epsilon=1e-6):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    return x.div(norm + epsilon)

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    b = values.shape[0]
    return values[torch.arange(b), indices.transpose(0, 1)].transpose(0, 1)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class ScaleNorm(nn.Module):
    def __init__(self, emb, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(1, requires_grad=True))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class WithNorm(nn.Module):
    def __init__(self, norm_class, emb, fn):
        super().__init__()
        self.emb = emb
        self.norm = norm_class(emb)
        self.fn = fn
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

class LSHAttention(nn.Module):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = False,
                  attend_across_buckets = False,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = torch.randn(rotations_shape, device=device).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, axis=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = torch.arange(self.n_hashes, device=device)
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape 
            buckets = torch.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def forward(self, qk, v):
        batch_size, seqlen, _ = qk.shape
        device = qk.device

        n_buckets = seqlen // self.bucket_size
        n_bins = n_buckets

        buckets = self.hash_vectors(n_buckets, qk)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        ticker = torch.arange(self.n_hashes * seqlen, device=device).unsqueeze(0)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = torch.reshape(st, (batch_size, self.n_hashes * n_bins, -1))
        bqk = torch.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, sqk.shape[-1]))
        bv = torch.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, sv.shape[-1]))
        bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, self.n_hashes * n_bins, -1))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = make_unit_length(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        bkv_buckets = look_one_back(bkv_buckets)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (bq.shape[-1] ** -0.5)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            dots[mask] = float('-inf')

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots[self_mask] = - 1e5

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots[bucket_mask] = float('-inf')

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % (self.n_hashes * n_bins)
            if not self._attend_across_buckets:
                locs1 = buckets * (self.n_hashes * n_bins) + locs1
                locs2 = buckets * (self.n_hashes * n_bins) + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, self.n_hashes * n_bins, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :]).sum(dim=-1)
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dots, bv)
        so = torch.reshape(bo, (batch_size, -1, bo.shape[-1]))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(Function):
            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, sticker)
                _, slogits_grad = sort_key_val(buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad

        o, logits = UnsortLogits.apply(so, slogits)

        if self.n_hashes == 1:
            out = o
        else:
            o = torch.reshape(o, (batch_size, self.n_hashes, seqlen, o.shape[-1]))
            logits = torch.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))
            probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdims=True))
            out = torch.sum(o * probs, dim=1)

        assert out.shape == v.shape
        return out

class LSHSelfAttention(nn.Module):
    def __init__(self, emb, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, random_rotations_per_head = False, **kwargs):
        super().__init__()
        self.heads = heads

        self.toqk = nn.Linear(emb, emb * heads)
        self.tov = nn.Linear(emb, emb * heads)
        self.unify_heads = nn.Linear(emb * heads, emb)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, causal=causal, random_rotations_per_head=random_rotations_per_head, **kwargs)

    def forward(self, x):
        b, t, e, h = *x.shape, self.heads
        assert t % self.bucket_size == 0, f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}'

        qk = self.toqk(x)
        v = self.tov(x)

        def merge_heads(v):
            return v.view(b, t, h, e).transpose(1, 2).reshape(b * h, t, e)

        def split_heads(v):
            return v.view(b, h, t, e).transpose(1, 2).contiguous()

        qk = merge_heads(qk)
        v = merge_heads(v)
        attn_out = self.lsh_attn(qk, v)
        out = split_heads(attn_out).view(b, t, h * e)

        return self.unify_heads(out)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, emb, mult = 4):
        super().__init__()
        self.emb = emb
        self.proj_in = nn.Linear(emb, emb * mult)
        self.proj_out = nn.Linear(emb * mult, emb)

    def forward(self, x):
        x = self.proj_in(x)
        x = F.gelu(x)
        x = self.proj_out(x)
        return x

# reformer lm

class Reformer(nn.Module):
    def __init__(self, emb, depth, max_seq_len, num_tokens = 10000, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, causal = False, weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False):
        super().__init__()
        self.emb = emb
        self.depth = depth
        self.token_emb = nn.Embedding(num_tokens, emb)
        self.pos_emb = nn.Embedding(max_seq_len, emb)

        get_attn = lambda: LSHSelfAttention(emb, heads, bucket_size, n_hashes, causal = causal, dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head)
        get_ff = lambda: FeedForward(emb)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()

            f = WithNorm(norm_type, emb, attn)
            g = WithNorm(norm_type, emb, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = Chunk(ff_chunks, g, along_dim = -2)

            blocks.append(ReversibleBlock(f, g, split_along_dim=-1))

        self.layers = ReversibleSequence(nn.ModuleList(blocks))
        self.to_logits = nn.Linear(emb, num_tokens)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = torch.cat([x, x], dim = -1)
        x = self.layers(x)
        x = torch.stack(x.chunk(2, dim=-1)).sum(dim=0)
        return self.to_logits(x)