import torch
from torch import nn
from torch import Tensor
from layers.basics import SigmoidRange, Transpose, get_activation_fn, RMSNorm
from layers.pos_encoding import positional_encoding
from layers.attention import MultiheadAttention
from typing import Optional
import torch.nn.functional as F


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x patch_num]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x patch_num]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
   

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, patch_num, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*patch_num
       
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                    
        """
        x: [bs x nvars x d_model x patch_num]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * patch_num)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x patch_num]
        output: tensor [bs x nvars x patch_num x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x patch_num x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x patch_num x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x patch_num x nvars x patch_len]
        return x


class BackboneEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False, pe='zeros', learn_pe=True, verbose=False):

        super().__init__()
        self.n_vars = c_in
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding          

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, patch_num, n_vars, _ = x.shape

        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, patch_num, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1,n_vars, patch_num, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0, 1, 3, 2)                                                   # z: [bs x nvars x d_model x num_patch]

        return z
   
   
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, norm='BatchNorm', attn_dropout=0.,
                 dropout=0., activation='gelu', res_attention=False, n_layers=1,
                 pre_norm=False, store_attn=False):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    
    # def forward(self, src:Tensor, prefix_d: torch.Tensor = None, prefix_r: torch.Tensor = None):
    def forward(self, src:Tensor, prefix_d: torch.Tensor = None):
        """
        src: tensor [bs x seq_len x d_model]
        prefix_d: tensor [n_layers x 2 x bs x domain_len x d_model]
        # prefix_r: tensor [n_layers x 2 x bs x resolution_len x d_model]
        """
        output = src
        scores = None
        # if prefix_d is None or prefix_r is None:
        if prefix_d is None:
            if self.res_attention:
                # for mod in self.layers: output, scores = mod(output, prefix_d, prefix_r, prev=scores)
                for mod in self.layers: output, scores = mod(output, prefix_d, prev=scores)
                return output
            else:
                # for mod in self.layers: output = mod(output, prefix_d, prefix_r)
                for mod in self.layers: output = mod(output, prefix_d)
                return output
        else:
            if self.res_attention:
                # for i, mod in enumerate(self.layers): output, scores = mod(output, prefix_d[i], prefix_r[i], prev=scores)
                for i, mod in enumerate(self.layers): output, scores = mod(output, prefix_d[i], prev=scores)
                return output
            else:
                # for i, mod in enumerate(self.layers): output = mod(output, prefix_d[i], prefix_r[i])
                for i, mod in enumerate(self.layers): output = mod(output, prefix_d[i])
                return output
   
   
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            # self.norm_attn = nn.LayerNorm(d_model)    
            self.norm_attn = RMSNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    # def forward(self, src:Tensor, prefix_d: torch.Tensor = None, prefix_r: torch.Tensor = None, prev:Optional[Tensor]=None):
    def forward(self, src:Tensor, prefix_d: torch.Tensor = None, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        prefix_d: tensor [2 x bs x domain_len x d_model]
        # prefix_r: tensor [2 x bs x resolution_len x d_model]
        """
       
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # if prefix_d is None or prefix_r is None:
        if prefix_d is None:
            ## Multi-Head attention
            if self.res_attention:
                src2, attn, scores = self.self_attn(src, src, src, prev)
            else:
                src2, attn = self.self_attn(src, src, src)
        else:
            # K = torch.cat([prefix_d[0], prefix_r[0], src], dim=1)
            # V = torch.cat([prefix_d[1], prefix_r[1], src], dim=1)
            K = torch.cat([prefix_d[0], src], dim=1)
            V = torch.cat([prefix_d[1], src], dim=1)
            ## Multi-Head attention
            if self.res_attention:
                src2, attn, scores = self.self_attn(src, K, V, prev)
            else:
                src2, attn = self.self_attn(src, K, V)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src    


class TowerEncoder(nn.Module):
    """
    x: tensor [bs x nvars x d_model x num_patch]
    prefix_d: tensor [n_layers x 2 x bs x domain_len x d_model]
    # prefix_r: tensor [n_layers x 2 x bs x resolution_len x d_model]
    """
    def __init__(self, d_model, n_heads, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
       
        super().__init__()
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers,
                                    store_attn=store_attn)
   
    # def forward(self, x, prefix_d: torch.Tensor = None, prefix_r: torch.Tensor = None):
    def forward(self, x, prefix_d: torch.Tensor = None):
        bs, nvars, d_model, num_patch = x.shape
        x = x.permute(0,1,3,2)
        x = x.reshape(bs, nvars * num_patch, d_model)
        # x = self.encoder(x, prefix_d, prefix_r)
        x = self.encoder(x, prefix_d)
        x = x.reshape(bs, nvars, num_patch, d_model)
        x = x.permute(0,1,3,2)
        return x
   

# class PrefixEncoder(torch.nn.Module):
#     '''
#     The torch.nn model to encode the prefix
#     input: tensor [bs x prefix_len]
#     output: tensor [bs x prefix_len x n_layers * 2 * d_model]
#     '''
#     def __init__(self, prefix_projection, num_virtual_tokens, token_dim, encoder_hidden_size, n_layers):
#         super().__init__()
#         self.prefix_projection = prefix_projection
#         if self.prefix_projection:
#             self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
#             # Use a two-layer MLP to encode the prefix
#             self.transform = nn.Sequential(
#                 nn.Linear(token_dim, encoder_hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(
#                     encoder_hidden_size,
#                     n_layers * 2 * token_dim,
#                 ),
#             )
#         else:
#             self.embedding = nn.Embedding(num_virtual_tokens, n_layers * 2 * token_dim)

#     def forward(self, prefix: torch.Tensor):
#         if self.prefix_projection:
#             prefix_tokens = self.embedding(prefix)
#             past_key_values = self.transform(prefix_tokens)
#         else:
#             past_key_values = self.embedding(prefix)
#         return past_key_values
   
   
class MixtrueExpertsLayer(torch.nn.Module):
    '''
    The torch.nn model to encode the prefix
    input: tensor [bs x seq_len x d_model] [bs x prefix_len]
    output: tensor [bs x prefix_len x n_layers * 2 * d_model]
    '''
    def __init__(self, prefix_projection, num_virtual_tokens, token_dim, encoder_hidden_size, n_layers):
        super().__init__()
        self.prefix_projection = prefix_projection
        # gating
        self.gate = nn.Linear(token_dim, num_virtual_tokens)
        if self.prefix_projection:
            self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
            # Use a two-layer MLP to encode the prefix
            self.transform = nn.Sequential(
                nn.Linear(token_dim, encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(
                    encoder_hidden_size,
                    n_layers * 2 * token_dim,
                ),
            )
        else:
            self.embedding = nn.Embedding(num_virtual_tokens, n_layers * 2 * token_dim)

    def forward(self, x: torch.Tensor, prefix: torch.Tensor):
        # router
        # bs, seq_len, d_model = x.shape
        # x = x.view(-1, d_model)
        # # router_logits -> (bs * seq_len, n_experts)
        router_logits = self.gate(x)          # bs x seq_len x d_model
        routing_weights = F.softmax(router_logits, dim=2, dtype=torch.float)
   
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            prefix_tokens = torch.einsum('...e,ed->...d', routing_weights, prefix_tokens)
            past_key_values = self.transform(prefix_tokens)
        else:
            prefix_tokens = self.embedding(prefix)
            past_key_values = torch.einsum('...e,ed->...d', routing_weights, prefix_tokens)
        return past_key_values


class IncontextEncoder(nn.Module):
    '''
    input:  x [bs x n_vars * (seq_len - size)//step+1 x d_model]
    out:    x [bs x nvars x pred_len]
    '''
    def __init__(self, d_model, n_heads, d_ff=None, norm='BatchNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers,
                                    store_attn=store_attn)
       
    def forward(self, x, prefix:torch.Tensor=None):
        bs, _, d_model= x.shape
        x = self.encoder(x, prefix)
        return x

