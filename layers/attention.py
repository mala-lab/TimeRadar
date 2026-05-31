import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        # complex domain
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        # self.scale = 0.02
        # self.sparsity_threshold = 0.01
        # self.w_q = nn.Parameter(self.scale * torch.randn(2, d_k * n_heads, d_k * n_heads))
        # self.rb_q = nn.Parameter(self.scale * torch.randn(d_k * n_heads))
        # self.ib_q = nn.Parameter(self.scale * torch.randn(d_k * n_heads))
        # self.w_k = nn.Parameter(self.scale * torch.randn(2, d_k * n_heads, d_k * n_heads))
        # self.rb_k = nn.Parameter(self.scale * torch.randn(d_k * n_heads))
        # self.ib_k = nn.Parameter(self.scale * torch.randn(d_k * n_heads))
        # self.w_v = nn.Parameter(self.scale * torch.randn(2, d_v * n_heads, d_k * n_heads))
        # self.rb_v = nn.Parameter(self.scale * torch.randn(d_v * n_heads))
        # self.ib_v = nn.Parameter(self.scale * torch.randn(d_v * n_heads))


        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        # complex domain
        # self.to_out_w = nn.Parameter(self.scale * torch.randn(2, d_model, d_model))
        # self.to_out_rb = nn.Parameter(self.scale * torch.randn(d_model))
        # self.to_out_ib = nn.Parameter(self.scale * torch.randn(d_model))
        # self.to_out = nn.Dropout(proj_dropout)

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        # complex domain
        # q_s_real = (
        #     F.linear(Q.real, self.w_q[0].T) - \
        #     F.linear(Q.imag, self.w_q[1].T) + \
        #     self.rb_q
        # )
        # q_s_imag = (
        #     F.linear(Q.imag, self.w_q[0].T) + \
        #     F.linear(Q.real, self.w_q[1].T) + \
        #     self.ib_q
        # )
        # q_s = torch.stack([q_s_real, q_s_imag], dim=-1)
        # # q_s = F.softshrink(q_s, lambd=self.sparsity_threshold)
        # q_s = torch.view_as_complex(q_s).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        # k_s_real = (
        #     F.linear(K.real, self.w_k[0].T) - \
        #     F.linear(K.imag, self.w_k[1].T) + \
        #     self.rb_k
        # )
        # k_s_imag = (
        #     F.linear(K.imag, self.w_k[0].T) + \
        #     F.linear(K.real, self.w_k[1].T) + \
        #     self.ib_k
        # )
        # k_s = torch.stack([k_s_real, k_s_imag], dim=-1)
        # # k_s = F.softshrink(k_s, lambd=self.sparsity_threshold)
        # k_s = torch.view_as_complex(k_s).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s_real = (
        #     F.linear(V.real, self.w_v[0].T) - \
        #     F.linear(V.imag, self.w_v[1].T) + \
        #     self.rb_v
        # )
        # v_s_imag = (
        #     F.linear(V.imag, self.w_v[0].T) + \
        #     F.linear(V.real, self.w_v[1].T) + \
        #     self.ib_v
        # )
        # v_s = torch.stack([v_s_real, v_s_imag], dim=-1)
        # # v_s = F.softshrink(v_s, lambd=self.sparsity_threshold)
        # v_s = torch.view_as_complex(v_s).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        # complex domain
        # output_real = (
        #     F.linear(output.real, self.to_out_w[0].T) - \
        #     F.linear(output.imag, self.to_out_w[1].T) + \
        #     self.to_out_rb
        # )
        # output_imag = (
        #     F.linear(output.imag, self.to_out_w[0].T) + \
        #     F.linear(output.real, self.to_out_w[1].T) + \
        #     self.to_out_ib
        # )
        # output = torch.stack([output_real, output_imag], dim=-1)
        # output = F.softshrink(output, lambd=self.sparsity_threshold)
        # output = torch.view_as_complex(output)
        # output = self.to_out(output)

        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights


   
class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
        # complex domain
        # attn_scores = torch.einsum("bhqd,bhdk->bhqk", q, k) * self.scale


        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        # complex domain
        # attn_weights = F.softmax(attn_scores.abs(), dim=-1)
        # attn_weights = self.attn_dropout(attn_weights)
        # attn_weights = torch.complex(attn_weights, torch.zeros_like(attn_weights))

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]
        # complex domain
        # output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        # out_real = torch.matmul(attn_weights, v.real)
        # out_imag = torch.matmul(attn_weights, v.imag)
        # output = torch.complex(out_real, out_imag)


        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights