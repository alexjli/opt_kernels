import torch
import taichi as ti

from opt_kernels.kernels.taichi_ipa import ipa_sdpa_fwd, ipa_sdpa_bwd


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
    ):
        # TODO: to make this more robust we should do a bunch of shape checks
        assert q.dim() == 4

        # get some important constants
        c_v = v.shape[-1]
        n_v_pts = v_pts.shape[-2]
        B, L_i, L_j, D_bias = z_out_bias.shape
        B, H, L_i, L_j = z_attn_bias.shape
        device = q.device

        # create the tensors for taichi kernel output
        out = torch.zeros((B, H, L_i, c_v), device=device)
        out_pts = torch.zeros((B, H, L_i, n_v_pts, 3), device=device)
        out_bias = torch.zeros((B, H, L_i, D_bias), device=device)
        lse_store = torch.zeros((B, H, L_i), device=device)

        ti.ad.no_grad(ipa_sdpa_fwd)(
            q=q.contiguous().detach(),
            q_pts=q_pts.contiguous().detach(),
            k=k.contiguous().detach(),
            k_pts=k_pts.contiguous().detach(),
            v=v.contiguous().detach(),
            v_pts=v_pts.contiguous().detach(),
            z_attn_bias=z_attn_bias.contiguous().detach(),
            z_out_bias=z_out_bias.contiguous().detach(),
            out=out,
            out_pts=out_pts,
            out_bias=out_bias,
            L=lse_store,
            pts_bias_scale=pts_bias_scale.contiguous().detach()
        )

        return out, out_pts, out_bias, lse_store

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs, *outputs)

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
            out,
            out_pts,
            out_bias,
            lse_store
        ] = ctx.saved_tensors

        q_grad = torch.zeros_like(q)
        k_grad = torch.zeros_like(k)
        v_grad = torch.zeros_like(v)
        q_pts_grad = torch.zeros_like(q_pts)
        k_pts_grad = torch.zeros_like(k_pts)
        v_pts_grad = torch.zeros_like(v_pts)
        z_attn_bias_grad = torch.zeros_like(z_attn_bias)
        z_out_bias_grad = torch.zeros_like(z_out_bias)
        pts_bias_scale_grad = torch.zeros_like(pts_bias_scale)

        ti.ad.no_grad(ipa_sdpa_bwd)(
            q=q.contiguous().detach(),
            q_pts=q_pts.contiguous().detach(),
            k=k.contiguous().detach(),
            k_pts=k_pts.contiguous().detach(),
            v=v.contiguous().detach(),
            v_pts=v_pts.contiguous().detach(),
            z_attn_bias=z_attn_bias.contiguous().detach(),
            z_out_bias=z_out_bias.contiguous().detach(),
            out=out.contiguous().detach(),
            out_pts=out_pts.contiguous().detach(),
            out_bias=out_bias.contiguous().detach(),
            L=lse_store.contiguous().detach(),
            pts_bias_scale=pts_bias_scale.contiguous().detach(),
            q_grad=q_grad,
            q_pts_grad=q_pts_grad,
            k_grad=k_grad,
            k_pts_grad=k_pts_grad,
            v_grad=v_grad,
            v_pts_grad=v_pts_grad,
            z_attn_bias_grad=z_attn_bias_grad,
            z_out_bias_grad=z_out_bias_grad,
            out_grad=out_grad.contiguous().detach(),
            out_pts_grad=out_pts_grad.contiguous().detach(),
            out_bias_grad=out_bias_grad.contiguous().detach(),
            pts_bias_scale_grad=pts_bias_scale_grad
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
            pts_bias_scale_grad,
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
):
    # TODO: this isn't great code style
    inputs = [
        t.contiguous()
        for t in
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
        ]
    ]

    out, out_pts, out_bias, _ = FusedIPAKernel.apply(
        *inputs
    )
    return out, out_pts, out_bias


if __name__ == '__main__':
    import torch
    import taichi as ti
    import numpy as np
    import time

    from opt_kernels.kernels.taichi_ipa import W_L

    device = 'cuda:0'
    ti.init(arch=ti.cuda)

    B = 4
    H = 8
    L = 128
    D = 32
    D_bias = 16
    qk_pts = 8
    v_pts = 12

    q = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    k = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    v = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    q_pts = torch.randn((B, H, L, qk_pts, 3), dtype=torch.float32, device=device)
    k_pts = torch.randn((B, H, L, qk_pts, 3), dtype=torch.float32, device=device)
    v_pts = torch.randn((B, H, L, v_pts, 3), dtype=torch.float32, device=device)
    z_attn_bias = torch.randn((B, H, L, L), dtype=torch.float32, device=device)
    z_out_bias = torch.randn((B, H, L, L, D_bias), dtype=torch.float32, device=device)
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

    print("native pytorch")

    start = time.time()
    w_C = np.sqrt(2/(9*q_pts.shape[-2]))
    attn = torch.einsum("...ik,...jk->...ij", q, k) * 1/np.sqrt(q.shape[-1])
    attn = attn + z_attn_bias
    pts_disp = q_pts[..., None, :, :] - k_pts[..., None, :, :, :]
    # print(pts_disp.squeeze())
    pts_dist = torch.sum(
        pts_disp ** 2,
        dim=-1
    )
    pts_bias = pts_dist.sum(dim=-1)
    attn = attn - pts_bias * pts_bias_scale[None, :, None, None] * w_C / 2
    attn = torch.softmax(W_L * attn, dim=-1)
    out = torch.einsum("...ij,...jk->...ik", attn, v)
    out_pts = torch.einsum("...ij,...jpk->...ipk", attn, v_pts)
    out_bias = torch.sum(
        attn[..., None] * z_out_bias,
        dim=-2
    )
    torch.autograd.backward(
        tensors=[out, out_pts, out_bias],
        grad_tensors=[torch.ones_like(out), torch.ones_like(out_pts), torch.ones_like(out_bias)],
    )
    end = time.time()
    print(end-start)

    for t in [
        q, q_pts,
        k, k_pts,
        v, v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale
    ]:
        t.grad = None

    print("taichi kernel")
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
    )
    torch.autograd.backward(
        tensors=[out, out_pts, out_bias],
        grad_tensors=[torch.ones_like(out), torch.ones_like(out_pts), torch.ones_like(out_bias)],
    )
    end = time.time()
    print(end-start)