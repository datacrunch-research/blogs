import torch


FP8_AMAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn

FP4_AMAX = 6.0
FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
# midpoints and the corresponding bins
# representable positives = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
thresholds = [
    (5.0, 0b0110), (3.5, 0b0101), (2.5, 0b0100),
    (1.75, 0b0011), (1.25, 0b0010), (0.75, 0b0001), (0.25, 0b0000),
]


# x shape: (M, N/16, 16)
# - convert each fp32 value into 4 bits along with sign
# - pack 8x4bits into 1xint32 value: (M, N/16, 2) i.e. 64 bits
# - final view to uint8 (i.e. 2xfp4): (M, N/16, 8) i.e. 64 / 8 
def cvt_1xfp32_2xfp4(x: torch.Tensor):
    assert x.dtype == torch.float32

    bits = x.view(torch.int32)
    sign_bit = (bits >> 31) & 0x1

    x_abs = x.abs()
    # prevent double counting with alternate <= and <
    other_bits = torch.full_like(x_abs, 0b0111, dtype=torch.int)
    for i, (m, code) in enumerate(thresholds):
        mask = x_abs <= m if i % 2 == 0 else x_abs < m
        other_bits = torch.where(mask, code, other_bits)

    # each fp32 now as e2m1 (pack 8xfp4 values into 1xint32)
    e2m1 = (sign_bit << 3) | other_bits

    # shape here becomes (M, N/16, 2) as 2x int32
    e2m1x2 = (
        e2m1[..., ::8]
        | (e2m1[..., 1::8] << 4)
        | (e2m1[..., 2::8] << 8)
        | (e2m1[..., 3::8] << 12)
        | (e2m1[..., 4::8] << 16)
        | (e2m1[..., 5::8] << 20)
        | (e2m1[..., 6::8] << 24)
        | (e2m1[..., 7::8] << 28)
    )
    # shape becomes (M, N/16, 8) after view
    # 64 bits / 8 bits, so each element is 2x e2m1
    return e2m1x2.view(FP4_DTYPE)


# nvfp4 needs two scaling factors
# Global encoding scale (dtype: float32):
#   s_enc = 6 * 448 / amax_x    -> from calibration
#   s_dec = 1 / s_enc
# Local encoding scale (per 16-block, dtype: fp8 e4m3):
#   s_decb = amax_b / 6
#   scales = e4m3(s_decb * s_enc) -> save this
#   s_encb = s_enc / scales.float()
# Quant:
#   xi = q(xi * s_encb)
# q here packs 1xfp32 to 8xfp4
def quant_nvfp4_torch(x: torch.Tensor, global_scale: torch.Tensor = None):
    assert x.shape[-1] % 16 == 0
    
    batch_dim = tuple(x.shape[:-1])
    # (..., N/16, 16)
    x_blocks_f32 = x.unflatten(-1, (-1, 16)).float()

    q_dtype, q_dtype_max = FP4_DTYPE, FP4_AMAX
    s_dtype, s_dtype_max = FP8_DTYPE, FP8_AMAX

    if global_scale is None:
        global_scale = FP4_AMAX * FP8_AMAX / x_blocks_f32.abs().amax()

    # (..., N/16)
    s_decb = x_blocks_f32.abs().amax(dim=-1) / q_dtype_max
    xs = (s_decb * global_scale).clamp(
        -s_dtype_max, s_dtype_max
    ).to(s_dtype)

    # (..., N/16, 1)
    s_encb = (global_scale / xs.float().clip(1e-12)).unsqueeze(-1)
    x_blocks_f32 = x_blocks_f32 * s_encb
    xq = cvt_1xfp32_2xfp4(x_blocks_f32).reshape(*batch_dim, -1)

    return xq, xs, global_scale


if __name__ == "__main__":
    shape = (512, 128)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.01

    xq, xs, global_scale = quant_nvfp4_torch(x)
    print(">> Quantized tensor:")
    print(xq)
    print(xq.shape)
    print(">> Blockwise scales")
    print(xs)
    print(xs.shape)
    print(">> Global scale:")
    print(global_scale)
