
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, attn_mask=None):
    """
    q: [b, h, Lq, d]
    k: [b, h, Lk, d]
    v: [b, h, Lk, d]
    attn_mask: 形状可广播到 [b, h, Lq, Lk] 的加性掩码；
               填 0 的位置正常，需屏蔽的位置应为 -inf（或一个足够大的负数）
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [b,h,Lq,Lk]
    if attn_mask is not None:
        scores = scores + attn_mask  # 加性掩码：被屏蔽位置加上 -inf
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [b,h,Lq,d]
    return out, attn


class MultiheadAttention(nn.Module):
    """
    同一个模块既可做 self-attention 也可做 cross-attention：
    - self-attn: forward(x) 或 forward(q=x, k=x, v=x)
    - cross-attn: forward(q=dec_x, k=enc_x, v=enc_x)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        # x: [b, L, d_model] -> [b, h, L, d_head]
        b, L, _ = x.shape
        x = x.view(b, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        # x: [b, h, L, d_head] -> [b, L, d_model]
        b, h, L, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, L, h * d)
        return x

    def forward(self, x=None, q=None, k=None, v=None, attn_mask=None):
        """
        用法：
          1) Self-Attention: forward(x=src)  或 forward(q=src, k=src, v=src)
          2) Cross-Attention: forward(q=dec, k=enc, v=enc)

        attn_mask: 可选，加性掩码，形状可广播到 [b, h, Lq, Lk]
        """
        # 兼容 self-attn 的简写
        if q is None and k is None and v is None:
            assert x is not None, "Self-attention 传入 x，或显式传 q/k/v"
            q = k = v = x

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 注意力
        out, attn = scaled_dot_product_attention(q, k, v, attn_mask)

        # 合并 + 输出映射
        out = self._merge_heads(out)
        out = self.w_o(out)
        return out, attn

# --------- 常见 Mask 的构造（可选） ---------
def make_causal_mask(Lq, Lk, device=None):
    """
    生成因果 mask（下三角可见，上三角为 -inf），形状 [1,1,Lq,Lk]
    """
    mask = torch.full((Lq, Lk), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf，下三角为 0
    return mask.unsqueeze(0).unsqueeze(0)

def make_key_padding_mask(key_padding: torch.Tensor):
    """
    key_padding: [b, Lk]，True 表示该位置是 PAD 需要屏蔽
    返回形状 [b,1,1,Lk] 的加性掩码（PAD 位置为 -inf）
    """
    mask = key_padding.unsqueeze(1).unsqueeze(2)  # [b,1,1,Lk]
    return mask.masked_fill(mask, float('-inf')).to(key_padding.dtype) * 0 + (~key_padding).float() * 0 + (key_padding.float() * float('-inf'))

# ------------------ 用法示例 ------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    b = 2
    Lx = 5
    Lz = 7
    d_model = 512
    h = 8

    # 两个序列：src（比如 encoder 输出），tgt（比如 decoder 当前层）
    src = torch.randn(b, Lx, d_model)
    tgt = torch.randn(b, Lz, d_model)

    mha = MultiheadAttention(d_model=d_model, num_heads=h)

    # 1) Self-Attention（如 Encoder 层或 Decoder 的自注意力）
    #    - 可选因果 mask（仅在 decoder 自注意力时用）
    causal_mask = make_causal_mask(Lz, Lz, device=tgt.device)  # decoder 自注意力常用
    out_self, attn_self = mha(x=tgt, attn_mask=causal_mask)    # Q=K=V=tgt

    # 2) Cross-Attention（如 Decoder 的 encoder-decoder 注意力）
    #    Q 来自 decoder 当前层输入，K/V 来自 encoder 输出
    out_cross, attn_cross = mha(q=tgt, k=src, v=src)           # 无需因果 mask，一般也不需要

    print(out_self.shape, attn_self.shape)   # [b, Lz, d_model], [b, h, Lz, Lz]
    print(out_cross.shape, attn_cross.shape) # [b, Lz, d_model], [b, h, Lz, Lx]





