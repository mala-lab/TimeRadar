import torch
from torch import nn
import numpy as np
from typing import Optional, Any, Tuple
from torch.autograd import Function
from torch.distributions.normal import Normal
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        hidden_dim = configs.hidden_dim
        self.repr_dim = configs.d_model
        depth = configs.depth
        bn_dims = configs.bn_dims
        k = configs.k
        max_iters = configs.max_iters
        # Patch
        self.patch_len = configs.patch_len
        if seq_len % self.patch_len:
            self.patch_num = (seq_len // self.patch_len) + 1
        else:
            self.patch_num = seq_len // self.patch_len
        # Encoder
        self.mask_mode = configs.mask_mode
        self.input_embed = nn.Linear(self.patch_len, hidden_dim)
        self.encoder = Encoder(
            patch_len=self.patch_len,
            patch_num=self.patch_num,
            output_dims=self.repr_dim,
            hidden_dims=hidden_dim,
            depth=depth,
            backbone="dilated_conv",
        )
        # BottleNeck
        self.adaptive_bottleneck = AdaptiveBottleNeck(
            seq_len=self.patch_num,
            seq_dim=self.repr_dim, # update
            repr_dim=self.repr_dim,
            bn_dims=bn_dims,
            k=k,
        )
        # Decoder
        self.decoder = MLP_Decoder(self.repr_dim, self.patch_len)
        self.adv_decoder = MLP_Decoder(self.repr_dim, self.patch_len)
        self.grl = WarmStartGradientReverseLayer(hi=0.5, max_iters=max_iters, auto_step=True)

    def forward(self, x, norm=0, mask_mode=None, copies=10):    # copies 原来是10
        if mask_mode is None:
            mask_mode = self.mask_mode
        B, T, dims = x.size()
        # 0.normalization
        if norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        # 1.channel independence
        x = x.permute(0, 2, 1) # b x c x t
        x = x.reshape(B*dims, T) # b*c x t
        # 2.do patch
        if T % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            padding = torch.zeros([B, (length - T)]).to(x.device) # update
            input = torch.cat([x, padding], dim=1) # b*c x patch_num*patch_len
        else:
            length = T
            input = x # b*c x t
        input_patch = input.unfold(dimension=-1, size=self.patch_len, step=self.patch_len) # b*c x patch_num x patch_len
        input_patch = self.input_embed(input_patch)  # b*c x patch_num x hidden_dims
        # 3.mask: generate copies
        if mask_mode=="c":
            assert copies%2==0, "The number of copies of symmetric mask must be an even number"
            mask_1 = torch.from_numpy(np.random.binomial(1, 0.5, size=(B*dims*(copies//2), self.patch_num))).to(torch.bool) 
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0) # b*c*copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0 # patch symmetry mask
        elif mask_mode=="random":
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B*dims*copies, self.patch_num))).to(torch.bool) # b*c*copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0
        elif mask_mode=="nomask":
            copies=1
        # 4.encoder
        repr = self.encoder(input_patch) # b*c*copies x patch_num x repr_dim
        # 5.adpBN
        repr = torch.reshape(repr, (-1, self.patch_num*self.repr_dim))
        repr, balance_loss = self.adaptive_bottleneck(repr, repr)
        repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))
        # 5.norm decoder
        out = self.decoder(repr) # b*c*copies x patch_num*patch_len
        out = out.reshape(copies, B*dims, T)
        out = out[:, :, :T].reshape(copies, B, dims, T)
        out = out.permute(0, 1, 3, 2) # copies x b x t x c
        # de-Normalization
        if norm: 
            out = out * stdev.unsqueeze(dim=0).repeat(copies, 1, 1, 1) + means.unsqueeze(dim=0).repeat(copies, 1, 1, 1)

        return out # copies x b x t x c

    def cal_anomaly_score(self, batch_x, batch_out_copies, anomaly_criterion=None, L=1):
        score = torch.var(batch_out_copies, dim=0)                  # b x t x c
        score = torch.mean(score, dim=-1)                           # b x t
        if anomaly_criterion is None:
            return score
        else:
            output_mean = torch.mean(batch_out_copies, dim=0)       # b x t x c
            recon_score = anomaly_criterion(output_mean, batch_x)
            score = L*score+(1-L)*torch.mean(recon_score, dim=-1)   # b x t
            return score


class Encoder(nn.Module):
    def __init__(self, patch_len, patch_num, output_dims=512, hidden_dims=64, depth=10, backbone="dilated_conv"):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.backbone = backbone
        if backbone=="dilated_conv":
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
            )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x hidden_dims x patch_num
        repr = self.repr_dropout(self.feature_extractor(x)) # B x output_dims x patch_num
        repr = repr.permute(0, 2, 1)
        return repr # B x patch_num x output_dims
       

class MLP_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims): 
        super().__init__()
        hidden_dim = int(input_dims*2)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dims, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dims),
        )
        self.flatten = nn.Flatten(-2)
    def forward(self, x): # b*c x patch_num x repr_dim
        x = self.net(x) # b*c x patch_num*patch_len
        x = self.flatten(x)
        return x
    

class Flatten_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(input_dims, output_dims),   
        )
    def forward(self, x): # b*c x patch_num x repr_dim
        x = self.net(x) # b*c x patch_num*patch_len
        return x


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) /
            (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class BottleNeck(nn.Module):
    """Create BottleNeck"""
    def __init__(self, seq_len, repr_dim, bn_dim):
        super(BottleNeck, self).__init__()
        self.seq_len = seq_len
        self.repr_dim = repr_dim
        self.net = nn.Sequential(
                        nn.Linear(repr_dim, bn_dim),
                        nn.GELU(),
                        nn.Linear(bn_dim, bn_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(bn_dim, repr_dim),
                    )       

    def forward(self, repr): # b*c x pN*repr_dim
        """
        compress the representation dimension and restore it.
        [batch_size, patch_num*repr_dim] -> [batch_size, patch_num*bn_dim] -> [batch_size, patch_num*repr_dim]

        Args:
            repr (tensor): [batch_size, patch_num*repr_dim]

        Returns:
            repr (tensor): [batch_size, patch_num*repr_dim]
        """
        res = repr
        repr = repr.reshape(-1, self.seq_len, self.repr_dim)
        repr = self.net(repr) 
        repr = repr.reshape(-1, self.seq_len*self.repr_dim)
        repr = res + repr
        return repr


class AdaptiveBottleNeck(nn.Module):
    """
    Adaptive BottleNeck impleted by using sMoE
    """
    def __init__(self, seq_len, seq_dim, repr_dim, bn_dims=[8,16,32,64], noisy_gating=True, k=3):
        super(AdaptiveBottleNeck, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_bottlenecks = len(bn_dims)
        self.k = k
        self.bottlenecks = nn.ModuleList([
            BottleNeck(seq_len, repr_dim, bn_dim) for bn_dim in bn_dims
        ])
        self.w_gate = nn.Parameter(torch.zeros(seq_len*seq_dim, self.num_bottlenecks), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(seq_len*seq_dim, self.num_bottlenecks), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_bottlenecks)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
        
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits   # B x bottleneck_num

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_bottlenecks), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_bottlenecks and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, router_input, repr, loss_coef=1e-2):
        """Args:
        router_input: tensor shape [batch_size, seq_len*seq_dim]
        repr: tensor shape [batch_size, seq_len*repr_dim]
        [loss_coef: a scalar - multiplier on load-balancing losses]

        Returns:
        repr: a tensor with shape [batch_size, seq_len*repr_dim].
        [extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.]
        """
        gates, load = self.noisy_top_k_gating(router_input, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_bottlenecks, gates)
        inputs = dispatcher.dispatch(repr) 
        gates = dispatcher.expert_to_gates()
        outputs = [self.bottlenecks[i](inputs[i]) for i in range(self.num_bottlenecks)]
        repr = dispatcher.combine(outputs)
        return repr, loss
    

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    

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

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=1,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)
