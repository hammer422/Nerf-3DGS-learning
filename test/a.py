

import torch
import torch.nn as nn
import math


def scaled_dot_product(Q, K, V, mask=None):
    # [b, head_nums, seq_len, d]
    d_k = K.size(-1)
    output = torch.matmul(Q, K.transpose(-1, -2))
    output = output / math.sqrt(d_k)
    if mask is not None:
        output = output + mask # mask -inf
    attention = torch.softmax(output, dim=-1)
    values = torch.matmul(attention, V.transpose(-1, -2))
    return values



class MultiHeadAttentionn(nn.Module):
    def __init__(self, head_nums, d_feature):
        super().__init__()

        self.head_nums = head_nums
        self.d_feature = d_feature
        assert d_feature % head_nums == 0

        self.w_q = nn.Linear(d_feature, d_feature)
        self.w_k = nn.Linear(d_feature, d_feature)
        self.w_v = nn.Linear(d_feature, d_feature)

        self.project = nn.Linear(d_feature, d_feature)

    def _split(self, x):
        # [b,seqlen,d] -> [b,headnums,seqlen,d//headnums]
        b, seqlen, _ = x.shape
        x = x.view(b, seqlen, self.head_nums, self.d_feature // self.head_nums).permute(0,1,2,3)
        return x

    def _merge(self, x):
        # [b,headnums,seqlen,d//headnums] -> [b,seqlen,d]
        b, _, seqlen, _ = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b,seqlen,-1)
        return x

    def forward(self, x, q=None, k=None, v=None, mask=None):
        if q is None and k is None and v is None:
            # self-attention
            # [b,seqlen, d]
            q = k = v = x
        
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self._split(q)
        k = self._split(k)
        v = self._split(v)

        output = scaled_dot_product(q, k, v, mask)

        output = self._merge(output)

        output = self.project(output)

        return output









