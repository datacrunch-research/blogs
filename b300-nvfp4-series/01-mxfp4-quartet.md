# Quartet: Native FP4 Training Can Be Optimal for Large Language Models

The compute requirements for training frontier LLMs have been doubling every few months. One key lever for reducing these costs is lower-precision computation: executing matrix multiplications at lower bit-widths yields near-linear gains in throughput and energy efficiency.

While FP8 training has become practical (as demonstrated by DeepSeek-V3), the question remains: can we go even lower? NVIDIA's Blackwell architecture introduces native support for MXFP4 operations, which can almost double arithmetic throughput over FP8 while cutting energy roughly in half.

The challenge: existing algorithms for end-to-end FP4 training face significant accuracy degradation. Methods like SwitchBack, Jetfire, and HALO either lose precision when training in FP4 or fall back to higher precision for selected operations.

This post is a deep dive into **Quartet**, a new approach for accurate, end-to-end FP4 training where all major computations (linear layers) occur in low precision.

- We'll start by examining Quartet's core abstraction: the `QuantizedLinear` module and how it orchestrates forward and backward quantization.
- Next, we'll dissect the backward computation schemes that enable full FP4 training with proper gradient flow.
- Then we'll trace through the quantizer implementations, from the base STE quantizers to the MXFP4-specific variants.
- We'll dive deep into the Triton kernel that fuses Hadamard transforms with MXFP4 quantization.
- Finally, we'll examine how all these pieces compose in a Llama-style model.

Code: https://github.com/IST-DASLab/Quartet

# Quartet Architecture Overview

Before diving into code, let's understand the high-level design. Quartet's key insight is that forward and backward passes require *different* quantization strategies:

- **Forward pass**: Minimize quantization error (MSE) using QuEST with deterministic Hadamard + RMSE-optimal clipping
- **Backward pass**: Maintain unbiased gradients using Stochastic Rounding with randomized Hadamard transforms

This asymmetry is implemented through a modular architecture:

```
QuantizedLinear
    ├── weight_quantizer      # QuEST for forward (FP4, deterministic Hadamard)
    ├── activation_quantizer  # QuEST for forward (FP4, deterministic Hadamard)
    ├── gradient_quantizer    # AlbertTseng for backward (FP4, stochastic rounding)
    └── backward_scheme       # Q(E)Q(Wt)t_Q(Et)Q(Xt)t (full re-quantization)
```

# QuantizedLinear: The Core Module

Source: `src/models/quantization/__init__.py`

The `QuantizedLinear` module extends PyTorch's `nn.Linear` with pluggable quantizers for weights, activations, and gradients:

```python
class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        gradient_quantizer=None,
        backward_scheme=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        if gradient_quantizer is None:
            gradient_quantizer = NoQuantizer()
        if backward_scheme is None:
            backward_scheme = EW_EtX_Scheme()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        self.backward_scheme = backward_scheme
        self.backward_scheme.g_quantizer = gradient_quantizer

    def forward(self, x):
        xq = self.activation_quantizer(x)
        wq = self.weight_quantizer(self.weight)
        return self.backward_scheme(
            xq,
            wq,
        )
```

The forward pass is clean:
1. Quantize activations: `xq = self.activation_quantizer(x)`
2. Quantize weights: `wq = self.weight_quantizer(self.weight)`
3. Dispatch to backward scheme: `self.backward_scheme(xq, wq)`

The backward scheme is responsible for both the forward GEMM and the custom backward computation. This separation allows different gradient quantization strategies without modifying the forward path.

# Backward Schemes: Controlling Gradient Flow

Source: `src/models/quantization/backward.py`

Quartet implements three backward schemes with increasing levels of gradient quantization:

## EW_EtX: No Gradient Quantization (Baseline)

```python
class EW_EtX_Scheme(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_quantizer = None

    def forward(self, x, w):
        return F.linear(x, w, None)
```

This is a pass-through: standard `F.linear` with PyTorch's default autograd. Gradients flow in full precision.

## QEW_QEtX: Quantize Error Only

```python
class QEW_QEtXFn(Function):
    @staticmethod
    def forward(ctx, x, w, g_quantizer):
        ctx.save_for_backward(x, w)
        ctx.g_quantizer = g_quantizer
        return F.linear(x, w, None)

    @staticmethod
    def backward(ctx, e):
        x, w = ctx.saved_tensors

        grad_x = F.linear(
            ctx.g_quantizer(e),
            w.T,
            None,
        ) # Q(E)W

        batch_seq_dim = math.prod(x.shape[:-1])
        grad_w = torch.einsum(
            "ib,bj->ij",
            ctx.g_quantizer(e.reshape(batch_seq_dim, -1).T.contiguous()),
            x.reshape(batch_seq_dim, -1),
        ) # Q(Et)X

        return grad_x, grad_w, None
```

The notation here:
- `E` is the error / output gradient (`dy`)
- `Q(E)W` means: quantize the error, then multiply by full-precision weights
- `Q(Et)X` means: quantize the transposed error, then multiply by full-precision activations

The backward computation:
- **DGrad** (gradient w.r.t. input): `grad_x = Q(E) @ W^T`
- **WGrad** (gradient w.r.t. weight): `grad_w = Q(E^T) @ X`

Note the `einsum` for WGrad handles the batch dimension collapse and transpose in one operation.

## QEQWtt_QEtQXtt: Full Re-quantization (Quartet's Choice)

```python
class QEQWtt_QEtQXttFn(Function):
    @staticmethod
    def forward(ctx, x, w, g_quantizer):
        ctx.save_for_backward(x, w)
        ctx.g_quantizer = g_quantizer
        return F.linear(x, w, None)

    @torch.compile
    @staticmethod
    def backward(ctx, e):
        x, w = ctx.saved_tensors

        ctx.g_quantizer.re_randomize()

        grad_x = F.linear(
            ctx.g_quantizer(e),
            ctx.g_quantizer(w.T.contiguous()),
            None,
        ) # Q(E)Q(W)

        batch_seq_dim = math.prod(x.shape[:-1])
        grad_w = torch.einsum(
            "ib,jb->ij",
            ctx.g_quantizer(e.reshape(batch_seq_dim, -1).T.contiguous()),
            ctx.g_quantizer(x.reshape(batch_seq_dim, -1).T.contiguous()),
        ) # Q(Et)Q(Xt)t

        return grad_x, grad_w, None
```

This is Quartet's recommended scheme. Key differences from `QEW_QEtX`:

1. **Re-randomization**: `ctx.g_quantizer.re_randomize()` generates a new random Hadamard transform between backward passes. This decorrelates quantization errors across the two GEMMs.

2. **Full quantization**: Both operands of each GEMM are quantized:
   - **DGrad**: `grad_x = Q(E) @ Q(W^T)`
   - **WGrad**: `grad_w = Q(E^T) @ Q(X^T)^T`

3. **`@torch.compile`**: The backward is JIT-compiled for performance.

The einsum in WGrad uses `"ib,jb->ij"` (note the `jb` instead of `bj`), which handles the double transpose more elegantly: we're computing `Q(E^T) @ Q(X^T)^T` without explicit transposes.

The backward scheme registry:
```python
BACKWARD_SCHEMES = {
    "EW_EtX": EW_EtX_Scheme,
    "Q(E)W_Q(Et)X": QEW_QEtX_Scheme,
    "Q(E)Q(Wt)t_Q(Et)Q(Xt)t": QEQWtt_QEtQXtt_Scheme,
}
```

# Base Quantizers and Optimal Scales

Source: `src/models/quantization/quantizers/base.py`

All quantizers inherit from `BaseQuantizer`:

```python
class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits

    def forward(self, x):
        raise NotImplementedError

    def re_randomize(self):
        pass


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x
```

A critical lookup table for RMSE-optimal scales when quantizing Gaussian-distributed data:

```python
OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,   # <- Used for FP4
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}
```

These values are precomputed to minimize mean squared error when quantizing standard normal distributions. For 4-bit quantization, the optimal scale is approximately 2.51 standard deviations.

# QuEST Quantizers: The Forward Pass Workhorses

Source: `src/models/quantization/quantizers/quest.py`

## STEQuantizer: Basic Straight-Through Estimator

```python
class STEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits)
        self.centered = centered

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
            xq = torch.round(x_clip / step) * step

        return x + (xq - x).detach()
```

The key operation: `x + (xq - x).detach()`

This is the straight-through estimator (STE) trick:
- Forward: returns `xq` (quantized)
- Backward: gradients flow through `x` unchanged (because `(xq - x).detach()` has zero gradient)

The scale computation uses per-row RMSE: `sqrt(mean(x**2))` multiplied by the optimal Gaussian scale factor.

## TrustQuantizer: Selective Gradient Flow

```python
class TrustQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, trust=None):
        super().__init__(bits, centered)

        # in terms of std
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()
```

The "trust" mechanism: only propagate gradients where quantization error is small.

```python
mask = (torch.abs(xq - x) <= std * self.trust).float()
return x * mask + (xq - x * mask).detach()
```

If `|xq - x| > std * trust`, the gradient for that element is zeroed. This prevents large quantization errors from corrupting the gradient signal.

## HadamardTrustQuantizer: QuEST with Full Hadamard Transform

```python
class HadamardTrustQuantizer(TrustQuantizer):
    def __init__(self, bits=4, trust=None, hadamard_dim=128):
        super().__init__(bits, True, trust)
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)

        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = (xq.view(-1, self.hadamard_dim) @ self.aux_matrix).view_as(x)

        grad_flow_output = ((x_had * mask).view(-1, self.hadamard_dim) @ self.aux_matrix).view_as(x)

        return grad_flow_output + (xq - grad_flow_output).detach()
```

The Hadamard workflow:
1. **Transform**: `x_had = x @ H^T` (reshape to blocks, apply Hadamard)
2. **Quantize**: scale, clip, round in the Hadamard domain
3. **Inverse transform**: `xq = xq @ H` (Hadamard is self-inverse up to scaling)

The `with torch.no_grad()` block ensures that quantization/clipping operations don't pollute the gradient computation.

The Hadamard matrix is created using the `fast_hadamard_transform` library:
```python
self.aux_matrix = hadamard_transform(
    torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
)
```

This applies the Hadamard transform to the identity matrix with normalization factor `1/sqrt(dim)`.

# MXFP4 Quantizers: Hardware-Aligned Implementation

Source: `src/models/quantization/quantizers/mxfp4.py`

These quantizers target the MXFP4 format: 32-element groups sharing an 8-bit exponent scale.

## QuestMXFP4Quantizer: For Forward Pass

```python
class QuestMXFP4QuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix):
        x_dequantized, mask = mxfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            return_clip_mask=True,
            stochastic_round=False,
            quest=True,
            gaussian_scale=2.92247856 / 6.0,
        )
        ctx.save_for_backward(hadamard_matrix, mask)
        ctx.x_shape = x.shape
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        hadamard_matrix, mask = ctx.saved_tensors
        grad_input = (grad_output * mask.to(grad_output.dtype)).view(-1, hadamard_matrix.shape[0]) @ hadamard_matrix.T
        return grad_input.view(ctx.x_shape), None, None
```

Key parameters:
- `quest=True`: Use variance-based (RMSE) scaling instead of max-based
- `stochastic_round=False`: Deterministic rounding for forward pass
- `gaussian_scale=2.92247856 / 6.0`: The optimal scale factor for FP4's grid

The backward path:
1. Mask gradients using the clipping mask from forward
2. Apply inverse Hadamard: `@ hadamard_matrix.T`

```python
class QuestMXFP4Quantizer(BaseQuantizer):
    def __init__(self, hadamard_dim=32, rerotate=None):
        super().__init__(4)

        self.hadamard_dim = hadamard_dim
        self.hadamard_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
        )
        self.rerotate = rerotate

    def forward(self, x):
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        return QuestMXFP4QuantizerFn.apply(x, self.hadamard_matrix)

    def re_randomize(self):
        if self.rerotate == "signs":
            self.hadamard_matrix = self.hadamard_matrix @ torch.diag(
                torch.randint(
                    0, 2, (self.hadamard_dim,),
                    device=self.hadamard_matrix.device,
                    dtype=self.hadamard_matrix.dtype
                ) * 2 - 1
            )
        elif self.rerotate == "O32":
            gaussian_matrix = torch.randn(self.hadamard_dim, self.hadamard_dim, device=self.hadamard_matrix.device, dtype=self.hadamard_matrix.dtype)
            svd = torch.linalg.svd(gaussian_matrix)
            self.hadamard_matrix = svd[0] @ svd[2]
        elif self.rerotate is None:
            pass
        else:
            raise ValueError(f"Invalid rerotate value: {self.rerotate}")
```

The `re_randomize()` method supports two modes:
- `"signs"`: Multiply Hadamard by a random diagonal sign matrix (cheap)
- `"O32"`: Generate a fresh random orthogonal matrix via SVD (expensive)

## AlbertTsengQuantizer: For Backward Pass

```python
class AlbertTsengQuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix, stochastic_round):
        x_dequantized, _ = mxfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            return_clip_mask=False,
            stochastic_round=stochastic_round,
            quest=False,
            gaussian_scale=3/4,
        )
        ctx.save_for_backward(hadamard_matrix)
        ctx.x_shape = x.shape
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        hadamard_matrix, = ctx.saved_tensors
        grad_input = grad_output.view(-1, hadamard_matrix.shape[0]) @ hadamard_matrix.T
        return grad_input.view(ctx.x_shape), None, None


class AlbertTsengQuantizer(BaseQuantizer):
    def __init__(self, hadamard_dim=32, stochastic=True, rerotate=None):
        super().__init__(4)

        self.hadamard_dim = hadamard_dim
        self.hadamard_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
        )
        self.rerotate = rerotate
        self.stochastic = stochastic

    def forward(self, x):
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        return AlbertTsengQuantizerFn.apply(x, self.hadamard_matrix, self.stochastic)
```

Key differences from QuestMXFP4:
- `quest=False`: Use max-based scaling (for better handling of outliers in gradients)
- `stochastic_round=True`: Unbiased rounding for gradient estimation
- `gaussian_scale=3/4`: Different scaling factor for max-based approach

# The MXFP4 Triton Kernel: Fused Hadamard + Quantization

Source: `src/models/quantization/quantizers/mxfp4_triton.py`

This is where the magic happens. A single Triton kernel fuses:
1. Hadamard transform
2. Group-wise scaling
3. FP4 quantization (deterministic or stochastic)
4. Dequantization
5. Optional clipping mask generation

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 32}),
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def mxfp4_forward_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    output_ptr,
    clip_mask_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    group_size: tl.constexpr,
    gaussian_scale: tl.constexpr,
    stochastic_round: tl.constexpr,
    seed: int,
    quest: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
```

The kernel is autotuned across 5 block size configurations, from 1024 to 16384 elements per block.

## Hadamard Transform

```python
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(hadamard_dim, hadamard_dim)

    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)

    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)
```

The Hadamard transform is implemented as a simple matrix multiply (`tl.dot`). The input is reshaped into `(num_groups, hadamard_dim)` blocks, each multiplied by the 32x32 Hadamard matrix.

## Group-wise Scaling: QuEST vs Max-based

```python
    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))

    # scale
    if quest:
        mean_squared = tl.sum(x_had_grouped * x_had_grouped, axis=-1, keep_dims=True) / group_size
        mean = tl.sum(x_had_grouped, axis=-1, keep_dims=True) / group_size
        std = tl.sqrt(mean_squared - mean * mean)
        scales = gaussian_scale * std + 1e-8
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)))
        x_had_scaled = x_had_grouped / shared_exps
    else:
        scales = tl.max(tl.abs(x_had_grouped), axis=-1, keep_dims=True)
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)) - 2) / (3/4)
        x_had_scaled = x_had_grouped / shared_exps
```

Two scaling modes:
- **QuEST** (`quest=True`): Compute per-group variance, scale by `gaussian_scale * std`
- **Max-based** (`quest=False`): Scale by `max(abs(x))` with a 3/4 factor

The `shared_exps = tl.exp2(tl.floor(tl.log2(scales)))` converts the scale to a power-of-two, which is required for MXFP4's E8M0 scale format.

## FP4 Quantization

The FP4 grid values are: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (and their negatives).

### Deterministic Rounding

```python
    else:
        x_fp4 = tl.where(
            x_had_scaled_abs > 5,
            6,
            tl.where(
                x_had_scaled_abs > 3.5,
                4,
                tl.where(
                    x_had_scaled_abs > 2.5,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.75,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.25,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.75,
                                1,
                                tl.where(
                                    x_had_scaled_abs > 0.25,
                                    0.5,
                                    0,
                                )
                            )
                        )
                    )
                )
            )
        ) * x_had_scaled_sign
```

Thresholds are midpoints between adjacent FP4 values: `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5}`.

### Stochastic Rounding

```python
    if stochastic_round:
        x_fp4_high = tl.where(
            x_had_scaled_abs > 4,
            6,
            tl.where(
                x_had_scaled_abs > 3,
                4,
                tl.where(
                    x_had_scaled_abs > 2,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.5,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.0,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.5,
                                1,
                                0.5,
                            )
                        )
                    )
                )
            )
        )

        x_fp4_low = tl.where(
            x_had_scaled_abs > 4,
            4,
            tl.where(
                x_had_scaled_abs > 3,
                3,
                tl.where(
                    x_had_scaled_abs > 2,
                    2,
                    tl.where(
                        x_had_scaled_abs > 1.5,
                        1.5,
                        tl.where(
                            x_had_scaled_abs > 1.0,
                            1.0,
                            tl.where(
                                x_had_scaled_abs > 0.5,
                                0.5,
                                0.0,
                            )
                        )
                    )
                )
            )
        )

        prob_up = (x_had_scaled_abs - x_fp4_low) / (x_fp4_high - x_fp4_low)
        sampled_prob = tl.rand(seed, offsets).reshape(BLOCK_SIZE // hadamard_dim, hadamard_dim)
        x_fp4 = tl.where(
            sampled_prob < prob_up,
            x_fp4_high,
            x_fp4_low,
        ) * x_had_scaled_sign
```

Stochastic rounding computes:
1. `x_fp4_low`: floor to nearest FP4 value
2. `x_fp4_high`: ceiling to nearest FP4 value
3. `prob_up`: probability of rounding up (linear interpolation)
4. `sampled_prob`: random uniform [0, 1) per element

The final value is `high` with probability `prob_up`, else `low`. This ensures `E[x_fp4] = x_had_scaled` (unbiased).

## Clipping Mask and Dequantization

```python
    if clip_mask_ptr is not None:
        tl.store(
            clip_mask_ptr + offsets,
            tl.reshape(x_had_scaled_abs < 6, (BLOCK_SIZE,)),
            mask=mask
        )

    # dequantize
    x_dequantized = x_fp4 * shared_exps

    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))

    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)
```

The clipping mask identifies values within the FP4 representable range (`< 6` after scaling). This is used by QuEST to selectively propagate gradients.

## Python Wrapper

```python
@torch.compiler.disable()
def mxfp4_forward_kernel_wrapper(
    x,
    hadamard_matrix,
    return_clip_mask=False,
    gaussian_scale=3/4,
    stochastic_round=False,
    quest=True,
):
    # Make sure inputs are contiguous
    x = x.contiguous()

    # Create output tensor
    output = torch.empty_like(x)
    if return_clip_mask:
        clip_mask = torch.empty_like(x, dtype=torch.bool)
    else:
        clip_mask = None

    if stochastic_round:
        seed = randint(0, 1000000)
    else:
        seed = None

    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch optimized kernel
    mxfp4_forward_kernel[grid](
        x_ptr=x,
        hadamard_matrix_ptr=hadamard_matrix,
        output_ptr=output,
        clip_mask_ptr=clip_mask,
        n_elements=n_elements,
        hadamard_dim=hadamard_matrix.shape[-1],
        group_size=32,
        gaussian_scale=gaussian_scale,
        stochastic_round=stochastic_round,
        seed=seed,
        quest=quest,
    )

    return output, clip_mask
```

Note `@torch.compiler.disable()`: this prevents torch.compile from tracing into the Triton kernel, which would cause issues.

# Hadamard Transform Utilities

Source: `src/quarot_utils/hadamard_utils.py`

The codebase includes support for arbitrary dimension Hadamard transforms, not just powers of 2:

```python
def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert (is_pow2(n // 172))
        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert (is_pow2(n // 156))
        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert (is_pow2(n // 140))
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    # ... more cases for 108, 60, 52, 36, 28, 40, 20, 12
    else:
        assert (is_pow2(n))
        K = 1

    return hadK, K
```

This allows applying Hadamard transforms to Llama's intermediate dimensions (which aren't always powers of 2) by using composite Hadamard matrices.

The recursive CPU implementation:
```python
def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()
```

And the CUDA-accelerated version:
```python
def matmul_hadU_cuda(X, hadK, K):
    n = X.shape[-1]
    if K == 1:
        return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0/torch.tensor(n).sqrt())
    input = X.view(-1, K, n // K)
    input = fast_hadamard_transform.hadamard_transform(input.contiguous(), 1.0/torch.tensor(n).sqrt())
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)
```

Random Hadamard generation for decorrelation:
```python
def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)
```

# Integration in Llama Models

Source: `src/models/llama.py`

## LlamaMLP: SwiGLU with Quantized Linear

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )
        self.hidden_dim = hidden_dim

        self.w12 = QuantizedLinear(
            config.n_embd,
            hidden_dim * 2,
            bias=False,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](
                **config.a_quant_kwargs
            ),
            gradient_quantizer=QUANTIZER_CLASSES[config.g_quant](**config.g_quant_kwargs),
            backward_scheme=BACKWARD_SCHEMES[config.backward_scheme](**config.backward_scheme_kwargs),
        )
        self.c_proj = QuantizedLinear(
            hidden_dim,
            config.n_embd,
            bias=False,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](
                **config.a_quant_kwargs
            ),
            gradient_quantizer=QUANTIZER_CLASSES[config.g_quant](**config.g_quant_kwargs),
            backward_scheme=BACKWARD_SCHEMES[config.backward_scheme](**config.backward_scheme_kwargs),
        )

    def forward(self, x):
        up, gate = self.w12(x).split(self.hidden_dim, dim=-1)
        return self.c_proj(nn.functional.silu(up) * gate)
```

The SwiGLU formulation: `c_proj(silu(up) * gate)` where `up` and `gate` come from a fused linear projection.

Each `QuantizedLinear` is instantiated with:
- `weight_quantizer`: From `config.w_quant` (e.g., `QuestMXFP4Quantizer`)
- `activation_quantizer`: From `config.a_quant` (e.g., `QuestMXFP4Quantizer`)
- `gradient_quantizer`: From `config.g_quant` (e.g., `AlbertTsengQuantizer`)
- `backward_scheme`: From `config.backward_scheme` (e.g., `Q(E)Q(Wt)t_Q(Et)Q(Xt)t`)

## LlamaAttention with RoPE

```python
class LlamaAttention(CausalSelfAttention):

    def forward(self, x, freqs_cis):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (B, nh, T, hs)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
```

The `self.c_attn` and `self.c_proj` are `QuantizedLinear` layers inherited from `CausalSelfAttention`. RoPE is applied in full precision after the quantized QKV projection.

# Training Configuration

Source: `main_setup.sh`

The default Quartet configuration for a 30M parameter model:

```bash
# Quantization configuration
export W_QUANT="QuestMXFP4Quantizer"
export W_BITS=4
export W_QUANT_KWARGS="{}"

export A_QUANT="QuestMXFP4Quantizer"
export A_BITS=4
export A_QUANT_KWARGS="{}"

export G_QUANT="AlbertTsengQuantizer"
export G_BITS=4
export G_QUANT_KWARGS="{\"stochastic\": true, \"rerotate\":\"signs\"}"

export BACKWARD_SCHEME="Q(E)Q(Wt)t_Q(Et)Q(Xt)t"
export BACKWARD_SCHEME_KWARGS="{}"
```

Breaking this down:
- **Weights & Activations**: `QuestMXFP4Quantizer` - deterministic Hadamard + RMSE-optimal scaling
- **Gradients**: `AlbertTsengQuantizer` with:
  - `stochastic=true`: Unbiased rounding
  - `rerotate="signs"`: Re-randomize Hadamard via sign flipping between backward passes
- **Backward Scheme**: `Q(E)Q(Wt)t_Q(Et)Q(Xt)t` - full re-quantization

The training launch:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --model llama \
    --compile \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}" \
    --g-quant ${G_QUANT} \
    --g-quant-kwargs "${G_QUANT_KWARGS}" \
    --backward-scheme ${BACKWARD_SCHEME} \
    --backward-scheme-kwargs "${BACKWARD_SCHEME_KWARGS}"
```

# The Training Loop

Source: `src/optim/base.py`

The core training iteration:
```python
while curr_iter <= cfg.iterations:
    # Train model
    t_start = time.perf_counter_ns()
    for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
        x, y = get_batch(train_reader, device=cfg.device)
        with type_ctx:
            with distributed_backend.get_context_for_microstep_forward(
                model=model,
                microstep_idx=microstep_idx,
                gradient_accumulation_steps=cfg.acc_steps,
            ):
                outputs = model(x, targets=y)

        loss = outputs["loss"] / cfg.acc_steps
        with type_ctx:
            loss.backward()
```

The quantization happens automatically:
1. `model(x, targets=y)` calls `QuantizedLinear.forward()`, which quantizes weights and activations
2. `loss.backward()` triggers the custom `backward()` functions in the backward scheme, which quantize gradients

# Putting It All Together: The Computation Flow

For a single forward-backward pass through a `QuantizedLinear` layer:

## Forward Pass
```
x (BF16) → QuestMXFP4Quantizer → [Hadamard → QuEST scale → FP4 quant → dequant] → xq (BF16)
W (BF16) → QuestMXFP4Quantizer → [Hadamard → QuEST scale → FP4 quant → dequant] → wq (BF16)
y = xq @ wq.T
```

## Backward Pass (Q(E)Q(Wt)t_Q(Et)Q(Xt)t)
```
# Re-randomize Hadamard
AlbertTsengQuantizer.re_randomize()  # flip signs

# DGrad
dy (BF16) → AlbertTsengQuantizer → [RandHadamard → max scale → SR FP4 quant → dequant] → dy_q
W.T (BF16) → AlbertTsengQuantizer → [RandHadamard → max scale → SR FP4 quant → dequant] → wt_q
grad_x = dy_q @ wt_q

# WGrad
dy.T (BF16) → AlbertTsengQuantizer → [RandHadamard → max scale → SR FP4 quant → dequant] → dyt_q
x.T (BF16) → AlbertTsengQuantizer → [RandHadamard → max scale → SR FP4 quant → dequant] → xt_q
grad_w = dyt_q @ xt_q.T
```

# QuTLASS: Production CUTLASS Kernels

Source: `notebooks/benchmark_mxfp4.ipynb` + https://github.com/IST-DASLab/qutlass

While the Triton kernels are useful for prototyping, production speedups require optimized CUTLASS implementations. Quartet releases these as the **QuTLASS** library, targeting RTX 5090 (SM100) and B200/B300.

## Kernel Suite Overview

QuTLASS provides four specialized kernel types:

```python
from qutlass import (
    fusedQuantizeMx,      # Forward: HT + QuEST quantization
    backward_t_bf16,      # Backward: Transpose + HT + quantization
    backward_qt_bf16,     # Backward: Dequant + Transpose + HT + quantization
    matmul_mxf4_bf16_tn,  # MXFP4 GEMM with scale factors
)
from qutlass.utils import to_blocked  # Scale factor layout transform
```

## CudaGemm: The Production Forward/Backward Implementation

The benchmark notebook shows the full production implementation:

```python
ALPHA_FWD = torch.tensor(1., device="cuda")
ALPHA_BWD = torch.tensor(1./9., device="cuda")  # Compensates for 3/4 scaling (16/9 = (4/3)^2)

class CudaGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, deterministic=True):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.deterministic = deterministic

        # Fused Hadamard + QuEST quantization for inputs
        input_hf_e2m1, input_hf_e8m0, input_hf_mask = fusedQuantizeMx_op(
            input.flatten(end_dim=-2),
            FORWARD_HADAMARD_MATRIX,
            return_mask=input.requires_grad,
        )

        # Fused Hadamard + QuEST quantization for weights
        weight_hf_e2m1, weight_hf_e8m0, weight_hf_mask = fusedQuantizeMx_op(
            weight,
            FORWARD_HADAMARD_MATRIX,
            return_mask=input.requires_grad,
        )

        ctx.save_for_backward(
            input_hf_e2m1, input_hf_e8m0, input_hf_mask,
            weight_hf_e2m1, weight_hf_e8m0, weight_hf_mask
        )

        # Transform scale factors to blocked layout for tcgen05.mma
        input_hf_scale_block = to_blocked(input_hf_e8m0, False)
        weight_hf_scale_block = to_blocked(weight_hf_e8m0, False)

        # MXFP4 GEMM with native scale factor support
        out = matmul_mxf4_bf16_tn_op(
            input_hf_e2m1,
            weight_hf_e2m1,
            input_hf_scale_block,
            weight_hf_scale_block,
            ALPHA_FWD,
        )
        return out.view(*input.shape[:-1], weight.size(-2))
```

Key observations:
- `fusedQuantizeMx` outputs: `e2m1` (FP4 values packed as uint8), `e8m0` (scales), `mask` (clipping mask)
- `to_blocked` transforms scale factors to the layout required by Blackwell's `tcgen05.mma` instruction
- The GEMM operates directly on quantized MXFP4 tensors with E8M0 scales

## Backward Pass with CUTLASS Kernels

```python
    @staticmethod
    def backward(ctx, grad_output):
        global BACKWARD_HADAMARD_MATRIX
        (input_hf_e2m1, input_hf_e8m0, input_hf_mask,
         weight_hf_e2m1, weight_hf_e8m0, weight_hf_mask) = ctx.saved_tensors

        # Re-randomize Hadamard for decorrelation
        if not ctx.deterministic:
            BACKWARD_HADAMARD_MATRIX = BACKWARD_HADAMARD_MATRIX * (
                torch.randint(0, 2, (32,), device=BACKWARD_HADAMARD_MATRIX.device,
                              dtype=BACKWARD_HADAMARD_MATRIX.dtype) * 2. - 1.
            )

        # === DGrad: grad_x = Q(E) @ Q(W^T) ===

        # Quantize gradient output
        grad_output_hb_e2m1, grad_output_hb_e8m0, _ = fusedQuantizeMx_op(
            grad_output.flatten(end_dim=-2),
            BACKWARD_HADAMARD_MATRIX,
            False,  # No mask needed for backward
        )

        # Fused: dequant weights -> transpose -> Hadamard -> quantize
        hft_weightt_hb_e2m1, hft_weightt_hb_e8m0 = backward_qt_bf16_op(
            weight_hf_e2m1, weight_hf_e8m0,
            BACKWARD_HADAMARD_MATRIX,
            ALPHA_FWD
        )

        # MXFP4 GEMM for DGrad
        grad_output_hb_scale_block = to_blocked(grad_output_hb_e8m0, False)
        hft_weightt_hb_scale_block = to_blocked(hft_weightt_hb_e8m0, False)
        grad_input_hf = matmul_mxf4_bf16_tn_op(
            grad_output_hb_e2m1,
            hft_weightt_hb_e2m1,
            grad_output_hb_scale_block,
            hft_weightt_hb_scale_block,
            ALPHA_BWD,  # 1/9 to compensate for 3/4 scaling
        )

        # Apply mask and inverse Hadamard
        input_mask_hf = _unpack_mask(input_hf_mask)
        grad_input = (
            (grad_input_hf.view(-1, 32) * input_mask_hf.view(-1, 32).to(grad_input_hf.dtype))
            @ FORWARD_HADAMARD_MATRIX.T
        ).view(*grad_output.shape[:-1], weight_hf_e2m1.size(-1) * 2)

        # === WGrad: grad_w = Q(E^T) @ Q(X^T)^T ===

        # Fused: transpose grad_output -> Hadamard -> quantize
        grad_outputt_hb_e2m1, grad_outputt_hb_e8m0 = backward_t_bf16_op(
            grad_output.flatten(end_dim=-2),
            BACKWARD_HADAMARD_MATRIX
        )

        # Fused: dequant input -> transpose -> Hadamard -> quantize
        hft_inputt_hb_e2m1, hft_inputt_hb_e8m0 = backward_qt_bf16_op(
            input_hf_e2m1, input_hf_e8m0,
            BACKWARD_HADAMARD_MATRIX,
            ALPHA_FWD
        )

        # MXFP4 GEMM for WGrad
        grad_outputt_hb_scale_block = to_blocked(grad_outputt_hb_e8m0, False)
        hft_inputt_hb_scale_block = to_blocked(hft_inputt_hb_e8m0, False)
        grad_weight_hf = matmul_mxf4_bf16_tn_op(
            grad_outputt_hb_e2m1,
            hft_inputt_hb_e2m1,
            grad_outputt_hb_scale_block,
            hft_inputt_hb_scale_block,
            ALPHA_BWD,
        )

        # Apply mask and inverse Hadamard
        weight_mask_hf = _unpack_mask(weight_hf_mask)
        grad_weight = (
            (grad_weight_hf.view(-1, 32) * weight_mask_hf.view(-1, 32).to(grad_weight_hf.dtype))
            @ FORWARD_HADAMARD_MATRIX.T
        ).view(grad_output.size(-1), weight_hf_e2m1.size(-1) * 2)

        return grad_input, grad_weight, None
```

## Specialized Backward Kernels

The backward pass requires three distinct fused operations:

### 1. `fusedQuantizeMx`: Hadamard + Quantization (same as forward)
```
Input: x [M, K] (BF16)
Output: x_e2m1 [M, K/2] (uint8 packed FP4), x_e8m0 [M, K/32] (E8M0 scales)
```

### 2. `backward_t_bf16`: Transpose + Hadamard + Quantization
```
Input: x [M, K] (BF16)
Output: xt_e2m1 [K, M/2] (uint8), xt_e8m0 [K, M/32] (E8M0)

Fuses: x.T -> Hadamard -> MXFP4 quantize
```

### 3. `backward_qt_bf16`: Dequant + Transpose + Hadamard + Re-quantization
```
Input: x_e2m1 [M, K/2], x_e8m0 [M, K/32], hadamard [32, 32], alpha
Output: xt_e2m1 [K, M/2], xt_e8m0 [K, M/32]

Fuses: MXFP4 dequant -> x.T -> Hadamard -> MXFP4 quantize

This is the critical kernel for Q(E)Q(Wt)t_Q(Et)Q(Xt)t backward scheme:
it takes already-quantized forward activations/weights and re-quantizes
them with the randomized backward Hadamard.
```

## Custom Op Registration for torch.compile

QuTLASS kernels are registered as custom ops with fake tensor implementations:

```python
@torch.library.custom_op("quartet::fusedQuantizeMx_op", mutates_args=())
def fusedQuantizeMx_op(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, return_mask: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if return_mask:
        return fusedQuantizeMx(x_flat, hadamard_matrix, return_mask=True)
    else:
        return fusedQuantizeMx(x_flat, hadamard_matrix, return_mask=False) + (None,)

@fusedQuantizeMx_op.register_fake
def _(x_flat, hadamard_matrix, return_mask):
    rows, cols = x_flat.shape[0], x_flat.shape[1] // 32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4

    xh_e2m1 = torch.empty(
        x_flat.shape[0], x_flat.shape[1] // 2, dtype=torch.uint8, device=x_flat.device
    )
    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.uint8, device=x_flat.device
    )
    clip_mask = torch.empty(
        *x_flat.shape[:-1], x_flat.size(-1) // 8, dtype=torch.uint8, device=x_flat.device
    ) if return_mask else None
    return xh_e2m1, xh_e8m0, clip_mask
```

The fake tensor registration enables `torch.compile` to trace through the custom ops with correct shapes and dtypes.

## MXFP4 GEMM with Scale Factors

The core GEMM kernel supports Blackwell's native scale factor multiplication:

```python
@torch.library.custom_op("quartet::matmul_mxf4_bf16_tn_op", mutates_args=())
def matmul_mxf4_bf16_tn_op(
    x: torch.Tensor, w: torch.Tensor,
    xs: torch.Tensor, ws: torch.Tensor,
    alpha: torch.Tensor
) -> torch.Tensor:
    return matmul_mxf4_bf16_tn(
        x.view(torch.uint8),
        w.view(torch.uint8),
        xs.view(torch.float8_e8m0fnu),
        ws.view(torch.float8_e8m0fnu),
        alpha
    )
```

This maps to Blackwell's `tcgen05.mma` instruction:
```
D = alpha * (A * SFA) @ (B * SFB)
```

Where `SFA` and `SFB` are the E8M0 scale factors applied along the K dimension.

## Benchmark Results from Notebook

The notebook includes comprehensive benchmarks on RTX 5090 (hidden_size=4096):

```
GPU Benchmark Results (ms):
Method          Forward    Total
---------------------------------------------
baseline        5.15       16.51
+cuda           varies     varies (see speedups)
+fp8            varies     varies
```

**Quartet vs FP8 Speedups** (forward / backward):
| Model Size | Forward Speedup | Backward Speedup |
|------------|-----------------|------------------|
| 100M       | 1.64x           | 0.55x            |
| 800M       | 1.94x           | 0.71x            |
| 3B         | 2.04x           | 0.86x            |
| 7B         | 2.05x           | 0.93x            |
| 22B        | 2.51x           | 1.07x            |
| 52B        | 2.14x           | 1.38x            |

**Quartet vs BF16 Speedups**:
| Model Size | Forward Speedup | Backward Speedup |
|------------|-----------------|------------------|
| 100M       | 3.46x           | 0.87x            |
| 800M       | 4.34x           | 1.38x            |
| 3B         | 4.69x           | 1.78x            |
| 7B         | 4.74x           | 1.93x            |
| 22B        | 4.76x           | 2.19x            |
| 52B        | 3.63x           | 2.71x            |

Key insight: Forward pass achieves ~2x over FP8, but backward has overhead from the re-quantization kernels. For larger models, backward speedup approaches and exceeds 1x vs FP8.

# Quartet vs TransformerEngine NVFP4: Key Differences

Both Quartet and TransformerEngine's NVFP4 recipe target FP4 training on Blackwell, but they take fundamentally different approaches. Here's a detailed comparison:

## Quantization Format

| Aspect | TransformerEngine NVFP4 | Quartet |
|--------|------------------------|---------|
| **Data Type** | NVFP4 (E2M1 + E4M3 scales + FP32 amax) | MXFP4 (E2M1 + E8M0 scales) |
| **Scale Hierarchy** | 2-level: FP8 blockwise + FP32 global amax | 1-level: E8M0 per 32-element group |
| **Block Size** | 16 elements (1D) or 16x16 (2D) | 32 elements (matches Hadamard dim) |

TransformerEngine uses a **2-level scaling** scheme with FP8 E4M3 blockwise scales plus a global FP32 amax, while Quartet uses **single-level E8M0** scales per group. This simplifies Quartet's implementation but requires the Hadamard transform to handle outliers.

## Forward Pass Quantization

| Aspect | TransformerEngine NVFP4 | Quartet |
|--------|------------------------|---------|
| **Activations** | 1D rowwise RTNE | QuEST (Hadamard + RMSE-optimal scaling) |
| **Weights** | 2D blockwise RTNE (16x16 blocks) | QuEST (Hadamard + RMSE-optimal scaling) |
| **Pre-conditioning** | None for forward | Deterministic Hadamard transform |

Quartet applies **Hadamard transform before quantization** to spread outliers, then uses **variance-based (RMSE-optimal) scaling** instead of max-based scaling. TransformerEngine relies on 2D block quantization for weights to capture local statistics without pre-conditioning.

## Backward Pass Quantization

| Aspect | TransformerEngine NVFP4 | Quartet |
|--------|------------------------|---------|
| **DGrad (∂x)** | SR rowwise quantization of ∂y | Stochastic rounding with randomized Hadamard |
| **WGrad (∂W)** | RHT + SR columnwise for both ∂y and x | Stochastic rounding with randomized Hadamard |
| **Re-randomization** | Same RHT for x and ∂y | Re-randomize Hadamard between DGrad and WGrad |

The key difference: **Quartet re-randomizes the Hadamard transform** between DGrad and WGrad computations using `re_randomize()`. TransformerEngine uses the same RHT throughout.

```python
# TransformerEngine: Same Hadamard for both operands
qx_rht = RHT(x)  # Same transform
qdy_rht = RHT(dy)  # Same transform
grad_w = qdy_rht.T @ qx_rht

# Quartet: Re-randomize between operations
ctx.g_quantizer.re_randomize()  # Flip Hadamard signs
grad_x = Q(dy) @ Q(W.T)  # First quantization
# Re-randomization decorrelates errors
grad_w = Q(dy.T) @ Q(x.T).T  # Second quantization with different Hadamard
```

## Backward Scheme

| Aspect | TransformerEngine NVFP4 | Quartet |
|--------|------------------------|---------|
| **DGrad** | Q(∂y) @ W (full-precision weights) | Q(∂y) @ Q(W^T) (both quantized) |
| **WGrad** | Q(∂y^T) @ Q(x^T) | Q(∂y^T) @ Q(x^T) |
| **Weight re-quantization** | Uses cached forward quantization | Re-quantizes with backward Hadamard |

TransformerEngine's DGrad uses **full-precision weights** (cached columnwise from forward), while Quartet **re-quantizes weights** with the randomized backward Hadamard via `backward_qt_bf16`.

```python
# TransformerEngine DGrad
grad_x = Q_sr(dy) @ W_cached  # W already columnwise quantized in forward

# Quartet DGrad (Q(E)Q(Wt)t scheme)
grad_x = Q_sr_rht(dy) @ Q_sr_rht(W.T)  # Both re-quantized with randomized Hadamard
```

## Kernel Architecture

| Aspect | TransformerEngine | Quartet/QuTLASS |
|--------|------------------|-----------------|
| **Implementation** | Custom CUDA kernels in CuTe/Cutlass | CUTLASS 3.9 with Triton prototypes |
| **Warp Specialization** | Yes (8 warps: 1 DMA, 1 MMA, 4 epilogue, 2 aux) | TBD (QuTLASS internals) |
| **Pipeline Stages** | 2 pipelines (DMA→MMA, MMA→Epilogue) | 2 stages (fused quant, GEMM) |
| **Persistence** | Persistent kernel | Standard grid launch |

TransformerEngine's `hadamard_transform_cast_fusion` kernel is a highly specialized **warp-specialized persistent kernel** with explicit pipeline management:

```
TransformerEngine kernel structure:
┌─────────────────────────────────────────┐
│ 8 warps per CTA:                        │
│   • 1 DMA warp: TMA loads               │
│   • 1 MMA warp: RHT via tcgen05.mma     │
│   • 4 Epilogue warps: quantize + store  │
│   • 2 auxiliary warps                   │
├─────────────────────────────────────────┤
│ Pipelines:                              │
│   • DMA → MMA: multi-stage TMA          │
│   • MMA → Epilogue: 4-stage accum       │
└─────────────────────────────────────────┘
```

QuTLASS separates concerns into **specialized fused kernels**:

```
QuTLASS kernel structure:
┌─────────────────────────────────────────┐
│ Separate kernels for each operation:    │
│   • fusedQuantizeMx: HT + quantize      │
│   • backward_t_bf16: transpose + HT + Q │
│   • backward_qt_bf16: dequant + T + HT  │
│   • matmul_mxf4_bf16_tn: scaled GEMM    │
└─────────────────────────────────────────┘
```

## Scale Factor Handling

| Aspect | TransformerEngine | Quartet |
|--------|------------------|---------|
| **Layout Transform** | `swizzle_scaling_factors` kernel | `to_blocked` utility |
| **Swizzle Pattern** | 128x4 → 32x16 reorder for cublasLt | Blackwell tcgen05.mma layout |
| **Padding** | Pad to 128 (dim 0) and 4 (dim 1) | Pad to 128 (rows) and 4 (cols/32) |

Both require **scale factor layout transformation** for hardware compatibility, but the specific swizzle patterns differ based on the GEMM implementation (cublasLt vs QuTLASS custom).

## Tensor Storage

```python
# TransformerEngine: NVFP4Tensor with explicit layout management
class NVFP4Tensor:
    _rowwise_data: torch.Tensor       # E2M1 packed
    _columnwise_data: torch.Tensor    # E2M1 packed (transposed)
    _rowwise_scale_inv: torch.Tensor  # E4M3 blockwise
    _columnwise_scale_inv: torch.Tensor
    _amax_rowwise: torch.Tensor       # FP32 global
    _amax_columnwise: torch.Tensor

# Quartet: Simpler MXFP4 representation
# e2m1: uint8 packed FP4 values
# e8m0: uint8 E8M0 scales (one per 32 elements)
# No explicit rowwise/columnwise distinction - re-quantize as needed
```

TransformerEngine **caches both rowwise and columnwise** quantized tensors from forward pass. Quartet **re-quantizes on demand** in backward.

## Theoretical Foundation

| Aspect | TransformerEngine | Quartet |
|--------|------------------|---------|
| **Optimization Goal** | Minimize numerical error | Optimize scaling law parameters |
| **Key Metrics** | Quantization MSE, gradient variance | eff_N (parameter efficiency), eff_D (data efficiency) |
| **Design Principle** | Match FP16 training loss | Accept controlled accuracy loss for speedup |

Quartet introduces **scaling law analysis** to justify FP4 training: rather than demanding lossless precision, it quantifies when FP4 is *optimal* given compute budget via `eff_N` and `eff_D` parameters.

## Summary Table

| Feature | TransformerEngine NVFP4 | Quartet |
|---------|------------------------|---------|
| Scale hierarchy | 2-level (FP8 + FP32 amax) | 1-level (E8M0) |
| Forward pre-conditioning | None (2D blocks for W) | Hadamard for all |
| Backward pre-conditioning | RHT for WGrad only | Randomized Hadamard for all |
| DGrad weight source | Cached from forward | Re-quantized |
| Hadamard re-randomization | No | Yes (between DGrad/WGrad) |
| Rounding (forward) | RTNE | RTNE (deterministic) |
| Rounding (backward) | Stochastic | Stochastic |
| Scaling method | Max-based (AbsMax) | RMSE-optimal (QuEST) |
| Kernel style | Warp-specialized persistent | Separate fused kernels |
| Theoretical basis | Empirical stability | Scaling law optimality |

# Experimental Results Summary

From the paper:

**Accuracy** (30M params, C4 validation loss at 400x tokens/param):
| Method | Loss | eff_N | eff_D |
|--------|------|-------|-------|
| LUQ-INT4 | 3.40 | 0.50 | 0.15 |
| Quartet | **3.21** | **0.64** | **0.94** |

Quartet requires ~15% fewer parameters and ~5x less data to match LUQ-INT4.

**Speedups** (RTX 5090):
- Forward: up to **2x** over FP8, **4x** over BF16
- Training: up to **1.6x** over FP8, **2.9x** over BF16
- Inference prefill (7B): **1.41x** over FP8

# Key Takeaways

1. **Asymmetric quantization**: Forward uses QuEST (minimize MSE), backward uses stochastic rounding (minimize bias).

2. **Re-randomization is critical**: The `Q(E)Q(Wt)t_Q(Et)Q(Xt)t` scheme re-randomizes the Hadamard transform between DGrad and WGrad to decorrelate quantization errors.

3. **Fused kernels matter**: The Triton kernel fuses Hadamard + scaling + quantization + dequantization in a single pass, avoiding memory round-trips.

4. **MXFP4 format constraints**: Group size of 32, power-of-two exponent scales. The implementation aligns Hadamard dimension (32) with the MXFP4 group size.

5. **Modular design**: Quantizers, backward schemes, and layers are cleanly separated, enabling easy experimentation.

# References

- Quartet paper: https://arxiv.org/abs/2505.14669
- Quartet training code: https://github.com/IST-DASLab/Quartet
- QuTLASS (CUTLASS kernels): https://github.com/IST-DASLab/qutlass
- fast_hadamard_transform: https://github.com/Dao-AILab/fast-hadamard-transform
