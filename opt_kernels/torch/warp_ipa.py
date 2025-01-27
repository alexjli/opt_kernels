import math
from functools import lru_cache

import torch
import warp as wp

from opt_kernels.kernels.warp_ipa import generate_fwd_kernel, generate_bwd_kernel, BLOCK_SIZE, INF

class ComputeCapabilityError(Exception):
    pass

_FWD_KERNEL_CACHE = {}
_BWD_KERNEL_CACHE = {}

class FusedIPAKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        q,
        q_pts,
        k,
        k_pts,
        v,
        v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale,
        n_qk_pts,
        n_v_pts,
    ):
        # TODO: to make this more robust we should do a bunch of shape checks
        # get some important constants
        B, H, L_i, D = q.shape
        D_bias = z_out_bias.shape[-1]
        device = q.device

        # create the tensors for warp kernel output
        out = torch.zeros((B, H, L_i, D), device=device, dtype=torch.float32)
        out_pts = torch.zeros((B, H * n_v_pts, L_i, 3), device=device, dtype=torch.float32)
        out_bias = torch.zeros((B, H, L_i, D_bias), device=device, dtype=torch.float32)
        lse_store = torch.zeros((B, H, L_i), device=device, dtype=torch.float32)
        m_store = torch.full((B, H, L_i), fill_value=-INF, dtype=torch.float32, device=device)

        wp_inputs = [
            q,
            q_pts,
            k,
            k_pts,
            v,
            v_pts,
            z_attn_bias,
            z_out_bias,
            out,
            out_pts,
            out_bias,
            lse_store,
            m_store,
            pts_bias_scale,
        ]
        # wp_inputs = [wp.from_torch(t.detach(), return_ctype=True) for t in wp_inputs]
        wp_inputs = [wp.from_torch(t.detach()) for t in wp_inputs]

        kernel_cache_key = (D, D_bias, n_qk_pts, n_v_pts)
        if kernel_cache_key in _FWD_KERNEL_CACHE:
            ipa_sdpa_fwd = _FWD_KERNEL_CACHE[kernel_cache_key]
        else:
            ipa_sdpa_fwd = generate_fwd_kernel(
                channel_dim=D,
                bias_dim=D_bias,
                n_qk_pts=n_qk_pts,
                n_v_pts=n_v_pts,
                dtype=float
            )
            _FWD_KERNEL_CACHE[kernel_cache_key] = ipa_sdpa_fwd

        wp.launch_tiled(
            ipa_sdpa_fwd,
            dim=(B, H, math.ceil(L_i / BLOCK_SIZE)),
            inputs=wp_inputs,
            block_dim=256
        )

        return out, out_pts, out_bias, lse_store

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(
            # we do this because save_for_backward can only store tensors
            # but we need n_qk_pts and n_v_pts to generate/load the kernel
            *inputs[:-2],
            torch.tensor(inputs[-2:]),
            *outputs
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, out_grad, out_pts_grad, out_bias_grad, lse_store_grad):
        # the lse_store_grad is not useful

        [
            q,
            q_pts,
            k,
            k_pts,
            v,
            v_pts,
            z_attn_bias,
            z_out_bias,
            pts_bias_scale,
            (n_qk_pts, n_v_pts),
            out,
            out_pts,
            out_bias,
            lse_store
        ] = ctx.saved_tensors
        n_qk_pts = int(n_qk_pts.item())
        n_v_pts = int(n_v_pts.item())

        q_grad = torch.zeros_like(q)
        k_grad = torch.zeros_like(k)
        v_grad = torch.zeros_like(v)
        q_pts_grad = torch.zeros_like(q_pts)
        k_pts_grad = torch.zeros_like(k_pts)
        v_pts_grad = torch.zeros_like(v_pts)
        z_attn_bias_grad = torch.zeros_like(z_attn_bias)
        z_out_bias_grad = torch.zeros_like(z_out_bias)
        pts_bias_scale_grad = torch.zeros_like(pts_bias_scale)[..., None, None]

        # get some important constants
        H = q.shape[1]
        D = q.shape[-1]
        B, L_i, L_j, D_bias = z_out_bias.shape

        kernel_cache_key = (D, D_bias, n_qk_pts, n_v_pts)
        if kernel_cache_key in _BWD_KERNEL_CACHE:
            ipa_sdpa_bwd = _BWD_KERNEL_CACHE[kernel_cache_key]
        else:
            ipa_sdpa_bwd = generate_bwd_kernel(
                channel_dim=D,
                bias_dim=D_bias,
                n_qk_pts=n_qk_pts,
                n_v_pts=n_v_pts,
                dtype=float
            )
            _BWD_KERNEL_CACHE[kernel_cache_key] = ipa_sdpa_bwd

        wp_inputs = [
            q,
            q_pts,
            k,
            k_pts,
            v,
            v_pts,
            z_attn_bias,
            z_out_bias,
            out,
            out_pts,
            out_bias,
            lse_store,
            pts_bias_scale,
            q_grad,
            q_pts_grad,
            k_grad,
            k_pts_grad,
            v_grad,
            v_pts_grad,
            z_attn_bias_grad,
            z_out_bias_grad,
            out_grad,
            out_pts_grad,
            out_bias_grad,
            pts_bias_scale_grad
        ]
        wp_inputs = [wp.from_torch(t.detach(), return_ctype=True) for t in wp_inputs]

        wp.launch_tiled(
            ipa_sdpa_bwd,
            dim=(B, H, math.ceil(L_i / BLOCK_SIZE)),
            inputs=wp_inputs,
            block_dim=256
        )

        return (
            q_grad,
            q_pts_grad,
            k_grad,
            k_pts_grad,
            v_grad,
            v_pts_grad,
            z_attn_bias_grad,
            z_out_bias_grad,
            pts_bias_scale_grad[..., 0, 0],
            None,
            None
        )

def fused_ipa_kernel(
    q,
    q_pts,
    k,
    k_pts,
    v,
    v_pts,
    z_attn_bias,
    z_out_bias,
    pts_bias_scale,
    n_qk_pts=8,
    n_v_pts=12,
):
    # do our checks
    device = q.device
    major, minor = torch.cuda.get_device_capability(device)
    if not (major >= 7):
        raise ComputeCapabilityError(f"fused_ipa_kernel requires compute_cap>7.0 but {device} has compute_cap={major}.{minor}")

    L = q.shape[-2]
    assert L % BLOCK_SIZE == 0, f"fused_ipa_kernel can only accept inputs which are multiples of BLOCK_SIZE={BLOCK_SIZE}"

    out, out_pts, out_bias, _ = FusedIPAKernel.apply(
        q,
        q_pts,
        k,
        k_pts,
        v,
        v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale,
        n_qk_pts,
        n_v_pts
    )
    return out, out_pts, out_bias


if __name__ == '__main__':
    import torch
    import warp as wp
    import numpy as np
    import time

    from opt_kernels.kernels.warp_ipa import W_L

    device = 'cuda:0'
    # wp.config.verify_cuda = True
    wp.init()

    B = 2
    H = 8
    L = 512
    D = 64
    D_bias = 16
    n_qk_pts = 8
    n_v_pts = 12

    q = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    k = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    v = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    q_pts = torch.randn((B, H * n_qk_pts, L, 3), dtype=torch.float32, device=device)
    k_pts = torch.randn((B, H * n_qk_pts, L, 3), dtype=torch.float32, device=device)
    v_pts = torch.randn((B, H * n_v_pts, L, 3), dtype=torch.float32, device=device)
    z_attn_bias = torch.randn((B, H, L, L), dtype=torch.float32, device=device)
    z_out_bias = torch.randn((B, L, L, D_bias), dtype=torch.float32, device=device)
    pts_bias_scale = torch.ones(H, dtype=torch.float32, device=device)


    for t in [
        q, q_pts,
        k, k_pts,
        v, v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale
    ]:
        t.requires_grad = True

    print("warp kernel")
    out, out_pts, out_bias = fused_ipa_kernel(
        q=q,
        q_pts=q_pts,
        k=k,
        k_pts=k_pts,
        v=v,
        v_pts=v_pts,
        z_attn_bias=z_attn_bias,
        z_out_bias=z_out_bias,
        pts_bias_scale=pts_bias_scale,
        n_qk_pts=n_qk_pts,
        n_v_pts=n_v_pts
    )
    torch.autograd.backward(
        tensors=[out, out_pts, out_bias],
        grad_tensors=[torch.ones_like(out), torch.ones_like(out_pts), torch.ones_like(out_bias)],
    )

    for t in [
        q, q_pts,
        k, k_pts,
        v, v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale
    ]:
        t.grad = None

    for t in [
        q, q_pts,
        k, k_pts,
        v, v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale
    ]:
        t.requires_grad = True

    start = time.time()
    out, out_pts, out_bias = fused_ipa_kernel(
        q=q,
        q_pts=q_pts,
        k=k,
        k_pts=k_pts,
        v=v,
        v_pts=v_pts,
        z_attn_bias=z_attn_bias,
        z_out_bias=z_out_bias,
        pts_bias_scale=pts_bias_scale,
        n_qk_pts=n_qk_pts,
        n_v_pts=n_v_pts
    )
    torch.autograd.backward(
        tensors=[out, out_pts, out_bias],
        grad_tensors=[torch.ones_like(out), torch.ones_like(out_pts), torch.ones_like(out_bias)],
    )
    end = time.time()
    print(end-start)
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad
    q_pts_grad = q_pts.grad
    k_pts_grad = k_pts.grad
    v_pts_grad = v_pts.grad
    z_attn_bias_grad = z_attn_bias.grad
    z_out_bias_grad = z_out_bias.grad
    pts_bias_scale_grad = pts_bias_scale.grad


    print("native pytorch")

    for t in [
        q, q_pts,
        k, k_pts,
        v, v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale
    ]:
        t.grad = None

    q_pts = q_pts.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    k_pts = k_pts.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    v_pts = v_pts.unflatten(1, (H, n_v_pts)).transpose(-2, -3)
    q_pts = q_pts.detach()
    q_pts.requires_grad = True
    k_pts = k_pts.detach()
    k_pts.requires_grad = True
    v_pts = v_pts.detach()
    v_pts.requires_grad = True

    # for idx, t in enumerate([
    #     q, q_pts,
    #     k, k_pts,
    #     v, v_pts,
    #     z_attn_bias,
    #     z_out_bias,
    #     pts_bias_scale
    # ]):
    #     try:
    #         t.requires_grad = True
    #     except Exception as e:
    #         print(idx)
    #         raise e


    start = time.time()
    w_C = np.sqrt(2/(9*q_pts.shape[-2]))
    attn = torch.einsum("...ik,...jk->...ij", q, k) * 1/np.sqrt(q.shape[-1])
    attn = attn + z_attn_bias
    pts_disp = q_pts[..., None, :, :] - k_pts[..., None, :, :, :]
    pts_dist = torch.sum(
        pts_disp ** 2,
        dim=-1
    )
    pts_bias = pts_dist.sum(dim=-1)
    # pts_bias *= 0

    attn = attn - pts_bias * pts_bias_scale[None, :, None, None] * w_C / 2
    s_ij = attn * W_L
    attn = torch.softmax(s_ij, dim=-1)
    torch_out = torch.einsum("...ij,...jk->...ik", attn, v)
    torch_out_pts = torch.einsum("...ij,...jpk->...ipk", attn, v_pts)
    torch_out_bias = torch.sum(
        attn[..., None] * z_out_bias[..., None, :, :, :],
        dim=-2
    )
    torch.autograd.backward(
        tensors=[torch_out, torch_out_pts, torch_out_bias],
        grad_tensors=[torch.ones_like(torch_out), torch.ones_like(torch_out_pts), torch.ones_like(torch_out_bias)],
    )
    end = time.time()
    print(end-start)

    print(out[0, 0])
    print(torch_out[0, 0])
    assert torch.isclose(out, torch_out, atol=1e-6, rtol=1e-4).all()
    out_pts = out_pts.unflatten(1, (H, n_v_pts)).transpose(-2, -3)
    assert torch.isclose(out_pts, torch_out_pts, atol=1e-6, rtol=1e-4).all()
    assert torch.isclose(out_bias, torch_out_bias, atol=1e-6, rtol=1e-4).all()
    print(v_grad[0, 0])
    print(v.grad[0, 0])
    assert torch.isclose(v_grad, v.grad, atol=1e-6, rtol=1e-4).all()
    v_pts_grad = v_pts_grad.unflatten(1, (H, n_v_pts)).transpose(-2, -3)
    assert torch.isclose(v_pts_grad, v_pts.grad, atol=1e-6, rtol=1e-4).all()
    assert torch.isclose(z_out_bias_grad, z_out_bias.grad, atol=1e-6, rtol=1e-4).all()
    assert torch.isclose(z_attn_bias_grad, z_attn_bias.grad, atol=1e-6, rtol=1e-4).all()
    q_pts_grad = q_pts_grad.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    assert torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6, rtol=1e-4).all(), (
        q_pts_grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6, rtol=1e-4)],
        q_pts.grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6, rtol=1e-4)]
    )
    k_pts_grad = k_pts_grad.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    assert torch.isclose(k_pts_grad, k_pts.grad, atol=1e-6, rtol=1e-4).all()
    assert torch.isclose(q_grad, q.grad, atol=1e-6, rtol=1e-4).all(), (
        q_grad[~torch.isclose(q_grad, q.grad, atol=1e-6, rtol=1e-4)],
        q.grad[~torch.isclose(q_grad, q.grad, atol=1e-6, rtol=1e-4)],
    )
    assert torch.isclose(k_grad, k.grad, atol=1e-6, rtol=1e-4).all()
    # # TODO: why is this error so large relative to everything else...
    assert torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3, rtol=1e-4).all(), (
        pts_bias_scale_grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3, rtol=1e-4)],
        pts_bias_scale.grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3, rtol=1e-4)]
    )
