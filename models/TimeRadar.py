import torch
from torch import nn
import numpy as np
from typing import Optional, Any, Tuple
from torch.autograd import Function
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch_frft.frft_module import frft, ifrft
import math


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        hidden_dim = configs.hidden_dim
        self.repr_dim = configs.d_model
        depth = configs.depth
        # Patch
        self.patch_len = configs.patch_len
        if seq_len % self.patch_len:
            self.patch_num = (seq_len // self.patch_len) + 1
        else:
            self.patch_num = seq_len // self.patch_len
        # Encoder
        self.mask_mode = configs.mask_mode
        # Patch Embedding
        self.real_input_embed = nn.Linear(self.patch_len, hidden_dim)
        self.imag_input_embed = nn.Linear(self.patch_len, hidden_dim)
        self.encoder = FractionalEncoder(
            patch_len=self.patch_len,
            patch_num=self.patch_num,
            output_dims=self.repr_dim,
            hidden_dims=hidden_dim,
            depth=depth,
            backbone="dilated_conv",
        )
        # Reconstruction head
        self.decoder = MLP_Decoder(self.repr_dim, self.patch_len)
        self.net = nn.Sequential(           
            nn.Linear(seq_len, self.repr_dim),
            nn.ReLU(),
            nn.Linear(self.repr_dim, 1),
            nn.Sigmoid(),   
        )

    def forward(self, x, norm=0, mask_mode=None, copies=10):
        if mask_mode is None:
            mask_mode = self.mask_mode
        bs, seq_len, nvars = x.size()
        # Normalization
        if norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        # Channel Independence
        x = x.permute(0, 2, 1) # bs x nvars x seq_len
        x = x.reshape(bs * nvars, seq_len) # bs * nvars x seq_len

        # Patching
        if seq_len % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            padding = torch.zeros([bs * nvars, (length - seq_len)]).to(x.device) # update
            input = torch.cat([x, padding], dim=1) # bs * nvars x patch_num * patch_len
        else:
            length = seq_len
            input = x # bs * nvars x seq_len
        
        order = self.net(input.mean(dim=0)).squeeze()
        # fix order
        # order = torch.tensor(0.1, dtype=torch.float32, device=input.device)
        # frft
        z = frft(input.to(dtype=torch.complex64), order, dim=1)
        # FFT
        # z = torch.fft.fft(input, dim=1, norm='ortho')
        z1 = z.real
        z2 = z.imag

        # Patch Embedding
        real_input_patch = z1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) # bs * nvars x patch_num x patch_len
        imag_input_patch = z2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) # bs * nvars x patch_num x patch_len
        z1 = self.real_input_embed(real_input_patch)                                         # [bs x patch_num * nvars x d_model]  
        z2 = self.imag_input_embed(imag_input_patch)

        # 3.mask: generate copies
        if mask_mode=="c":
            mask_1 = (torch.rand((copies // 2) * bs * nvars, self.patch_num, device=z1.device) < 0.5)
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0) # [bs * copies x patch_num]
            mask = mask.unsqueeze(-1)       # [copies * bs, patch_num, 1]
            z1 = z1.unsqueeze(0).expand(copies, -1, -1, -1).reshape(copies * bs * nvars, -1, z1.size(-1)).masked_fill(mask, 0.)
            z2 = z2.unsqueeze(0).expand(copies, -1, -1, -1).reshape(copies * bs * nvars, -1, z2.size(-1)).masked_fill(mask, 0.)
        elif mask_mode=="random":
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(bs * nvars * copies, self.patch_num))).to(torch.bool) # bs * nvars * copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0
        elif mask_mode=="nomask":
            copies=1

        # Encoder
        repr = self.encoder(z1, z2, order) # bs * nvars * copies x patch_num x repr_dim

        # Reconstruction head
        out = self.decoder(repr) # bs * c * copies x patch_num * patch_len

        out = ifrft(out, order, dim=-1)
        # out = torch.fft.ifft(out, n=T, dim=1, norm="ortho")
        out = torch.abs(out)

        out = out.reshape(copies, bs * nvars, seq_len)
        out = out[:, :, :seq_len].reshape(copies, bs, nvars, seq_len)
        out = out.permute(0, 1, 3, 2) # copies x bs x seq_len x nvars
        # Denormalization
        if norm: 
            out = out * stdev.unsqueeze(dim=0).repeat(copies, 1, 1, 1) + means.unsqueeze(dim=0).repeat(copies, 1, 1, 1)

        return repr, out # copies x bs x seq_len x nvars

    def cal_anomaly_score(self, batch_x, batch_out_copies, anomaly_criterion=None, L=1):
        score = torch.var(batch_out_copies, dim=0)                  # bs x seq_len x nvars
        score = torch.mean(score, dim=-1)                           # bs x seq_len
        if anomaly_criterion is None:
            return score
        else:
            output_mean = torch.mean(batch_out_copies, dim=0)       # bs x seq_len x nvars
            recon_score = anomaly_criterion(output_mean, batch_x)
            score = L * score + (1 - L) * torch.mean(recon_score, dim=-1)   # bs x seq_len
            return score

    

class FractionalEncoder(nn.Module):
    def __init__(self, patch_len, patch_num, output_dims=512, hidden_dims=64, depth=10, backbone="dilated_conv"):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.backbone = backbone
        if backbone=="dilated_conv":
            self.feature_extractor = FractionalDilatedConvEncoder(
                hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
            )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, xr, xi, alpha=None):
        xr = xr.permute(0, 2, 1)  # bs x hidden_dims x patch_num
        xi = xi.permute(0, 2, 1)  # bs x hidden_dims x patch_num
        xr, xi = self.feature_extractor(xr, xi, alpha)
        xr = self.repr_dropout(xr) # bs x output_dims x patch_num
        xi = self.repr_dropout(xi)
        repr = torch.complex(xr, xi)
        repr = repr.permute(0, 2, 1)
        return repr # bs x patch_num x output_dims
       

class MLP_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims): 
        super().__init__()
        hidden_dim = int(input_dims*2)
        self.flatten = nn.Flatten(-2)
        self.scale = 0.02
        self.sparsity_threshold = 0.01
        self.w1 = nn.Parameter(self.scale * torch.randn(2, input_dims, hidden_dim))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, hidden_dim, hidden_dim))
        self.w3 = nn.Parameter(self.scale * torch.randn(2, hidden_dim, output_dims))

        self.rb1 = nn.Parameter(self.scale * torch.randn(hidden_dim))
        self.ib1 = nn.Parameter(self.scale * torch.randn(hidden_dim))

        self.rb2 = nn.Parameter(self.scale * torch.randn(hidden_dim))
        self.ib2 = nn.Parameter(self.scale * torch.randn(hidden_dim))

        self.rb3 = nn.Parameter(self.scale * torch.randn(output_dims))
        self.ib3 = nn.Parameter(self.scale * torch.randn(output_dims))
    
    def net(self, x):
        o1_real = F.gelu(
            F.linear(x.real, self.w1[0].T) - \
            F.linear(x.imag, self.w1[1].T) + \
            self.rb1
        )

        o1_imag = F.gelu(
            F.linear(x.imag, self.w1[0].T) + \
            F.linear(x.real, self.w1[1].T) + \
            self.ib1
        )

        o2_real = F.gelu(
            F.linear(o1_real, self.w2[0].T) - \
            F.linear(o1_imag, self.w2[1].T) + \
            self.rb2
        )

        o2_imag = F.gelu(
            F.linear(o1_imag, self.w2[0].T) + \
            F.linear(o1_real, self.w2[1].T) + \
            self.ib2
        )
        o3_real = (
            F.linear(o2_real, self.w3[0].T) - \
            F.linear(o2_imag, self.w3[1].T) + \
            self.rb3
        )

        o3_imag = (
            F.linear(o2_imag, self.w3[0].T) + \
            F.linear(o2_real, self.w3[1].T) + \
            self.ib3
        )
        y = torch.stack([o3_real, o3_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y
    
    def forward(self, x): # bs * nvars x patch_num x repr_dim
        x = self.net(x) # bs * nvars x patch_num*patch_len
        x = self.flatten(x)
        return x
    

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

        # Bottleneck
        self.down = nn.Conv1d(2 * in_channels, in_channels, kernel_size=1, bias=True)
        self.up   = nn.Conv1d(out_channels, 2 * out_channels, kernel_size=1, bias=True)

    def forward(self, xr, xi):
        if self.projector is None:
            rr, ri = xr, xi
        else:
            rr, ri = self.projector(xr), self.projector(xi)
        # concat along hidden dim
        x = torch.cat([xr, xi], dim=1)
        x = self.down(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.up(x)
        xr, xi = x.chunk(2, dim=1)
        return xr + rr, xi + ri
    

class FractionalConvBlock(nn.Module):
    def __init__(self, conv_block: nn.Module):
        super().__init__()
        self.conv_block = conv_block

    def _complex_mul(self, ar, ai, br, bi):
        # (ar + j ai) * (br + j bi)
        cr = ar * br - ai * bi
        ci = ar * bi + ai * br
        return cr, ci
    
    def _make_chirp(self, patch_num, alpha, device, dtype):
        # discrete position index
        n = torch.arange(patch_num, device=device, dtype=dtype)
        n = n - (patch_num - 1) / 2.0  # align center

        eps = 1e-4
        if alpha.dim() == 0:
            alpha = alpha[None]  
        alpha = alpha.to(device=device, dtype=dtype).view(-1, 1, 1)

        sin_a = torch.sin(alpha).clamp_min(eps)
        cot_a = torch.cos(alpha) / sin_a  # cot(alpha)

        # phase
        phase = math.pi * cot_a * (n.view(1, 1, -1) ** 2) / float(patch_num)

        chirp_r = torch.cos(phase)
        chirp_i = torch.sin(phase)
        return chirp_r, chirp_i

    def forward(self, xr, xi, alpha):
        # xr, xi: [bs * nvars * seq_len]
        bs, d_model, patch_num = xr.shape
        chirp_r, chirp_i = self._make_chirp(patch_num, alpha, xr.device, xr.dtype)  # 

        # modulation x * chirp
        xmr, xmi = self._complex_mul(xr, xi, chirp_r, chirp_i)

        # Original ConvBlock
        yr, yi = self.conv_block(xmr, xmi)

        # Inversion modulation chirp
        ydr, ydi = self._complex_mul(yr, yi, chirp_r, -chirp_i)
        return ydr, ydi


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=1,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, xr, xi):
        for blk in self.blocks:
            xr, xi = blk(xr, xi)
        return xr, xi
    

class FractionalDilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=1,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])
        # self.fblocks = nn.ModuleList([FractionalConvBlock(b) for b in self.blocks])

    def _complex_mul(self, ar, ai, br, bi):
        # (ar + j ai) * (br + j bi)
        cr = ar * br - ai * bi
        ci = ar * bi + ai * br
        return cr, ci
    
    def _make_chirp(self, patch_num, alpha, device, dtype):
        # discrete position index
        n = torch.arange(patch_num, device=device, dtype=dtype)
        n = n - (patch_num - 1) / 2.0  # align center

        eps = 1e-4
        if alpha.dim() == 0:
            alpha = alpha[None]  
        alpha = alpha.to(device=device, dtype=dtype).view(-1, 1, 1)

        sin_a = torch.sin(alpha).clamp_min(eps)
        cot_a = torch.cos(alpha) / sin_a  # cot(alpha)

        # phase
        phase = math.pi * cot_a * (n.view(1, 1, -1) ** 2) / float(patch_num)

        chirp_r = torch.cos(phase)
        chirp_i = torch.sin(phase)
        return chirp_r, chirp_i
    

    def forward(self, xr, xi, alpha):
        patch_num = xr.shape[-1]
        # make chirp
        chirp_r, chirp_i = self._make_chirp(patch_num, alpha, xr.device, xr.dtype)
        # modulate 
        xr, xi = self._complex_mul(xr, xi, chirp_r, chirp_i)

        # for fblk in self.fblocks:
        #     xr, xi = fblk(xr, xi, alpha)

        # plain conv blocks
        for blk in self.blocks:
            xr, xi = blk(xr, xi)

        # demodulate
        xr, xi = self._complex_mul(xr, xi, chirp_r, -chirp_i)
        return xr, xi

