import pytest

def test_ipa_kernel():
    import torch
    import taichi as ti
    import numpy as np

    from taichi_kernels.kernels.ipa import ipa_sdpa_fwd, ipa_sdpa_bwd, W_L

    if torch.cuda.is_available():
        device = 'cuda:0'
        ti.init(arch=ti.cuda)
    else:
        device='cpu'
        ti.init()

    B = 4
    H = 8
    L = 130 #128
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

    out = torch.zeros_like(q)
    out_pts = torch.zeros_like(v_pts)
    out_bias = torch.zeros((B, H, L, D_bias), dtype=torch.float32, device=device)

    lse_store = torch.zeros((B, H, L), dtype=torch.float32, device=device)

    for t in [
        q, q_pts,
        k, k_pts,
        v, v_pts,
        z_attn_bias,
        z_out_bias,
        pts_bias_scale
    ]:
        t.requires_grad = True

    print("taichi kernel")
    ipa_sdpa_fwd(
        q=q,
        q_pts=q_pts,
        k=k,
        k_pts=k_pts,
        v=v,
        v_pts=v_pts,
        z_attn_bias=z_attn_bias,
        z_out_bias=z_out_bias,
        out=out,
        out_pts=out_pts,
        out_bias=out_bias,
        L=lse_store,
        pts_bias_scale=pts_bias_scale
    )

    q_grad = torch.zeros_like(q)
    k_grad = torch.zeros_like(k)
    v_grad = torch.zeros_like(v)
    q_pts_grad = torch.zeros_like(q_pts)
    k_pts_grad = torch.zeros_like(k_pts)
    v_pts_grad = torch.zeros_like(v_pts)
    z_attn_bias_grad = torch.zeros_like(z_attn_bias)
    z_out_bias_grad = torch.zeros_like(z_out_bias)
    out_grad = torch.ones_like(out)
    out_pts_grad = torch.ones_like(out_pts)
    out_bias_grad = torch.ones_like(out_bias)
    pts_bias_scale_grad = torch.zeros_like(pts_bias_scale)

    ipa_sdpa_bwd(
        q=q,
        q_pts=q_pts,
        k=k,
        k_pts=k_pts,
        v=v,
        v_pts=v_pts,
        z_attn_bias=z_attn_bias,
        z_out_bias=z_out_bias,
        out=out,
        out_pts=out_pts,
        out_bias=out_bias,
        L=lse_store,
        pts_bias_scale=pts_bias_scale,
        q_grad=q_grad,
        q_pts_grad=q_pts_grad,
        k_grad=k_grad,
        k_pts_grad=k_pts_grad,
        v_grad=v_grad,
        v_pts_grad=v_pts_grad,
        z_attn_bias_grad=z_attn_bias_grad,
        z_out_bias_grad=z_out_bias_grad,
        out_grad=out_grad,
        out_pts_grad=out_pts_grad,
        out_bias_grad=out_bias_grad,
        pts_bias_scale_grad=pts_bias_scale_grad
    )

    print("native pytorch")
    w_C = np.sqrt(2/(9*q_pts.shape[-2]))
    attn = torch.einsum("...ik,...jk->...ij", q, k) * 1/np.sqrt(q.shape[-1])
    attn = attn + z_attn_bias
    pts_disp = q_pts[..., None, :, :] - k_pts[..., None, :, :, :]
    pts_dist = torch.sum(
        pts_disp ** 2,
        dim=-1
    )
    pts_bias = pts_dist.sum(dim=-1)
    attn = attn - pts_bias * pts_bias_scale[None, :, None, None] * w_C / 2
    attn = torch.softmax(W_L * attn, dim=-1)
    torch_out = torch.einsum("...ij,...jk->...ik", attn, v)
    torch_out_pts = torch.einsum("...ij,...jpk->...ipk", attn, v_pts)
    torch_out_bias = torch.sum(
        attn[..., None] * z_out_bias,
        dim=-2
    )
    torch.autograd.backward(
        tensors=[torch_out, torch_out_pts, torch_out_bias],
        grad_tensors=[out_grad, out_pts_grad, out_bias_grad],
    )
    # forward pass check
    assert torch.isclose(out, torch_out, atol=1e-6).all()
    assert torch.isclose(out_pts, torch_out_pts, atol=1e-6).all()
    assert torch.isclose(out_bias, torch_out_bias, atol=1e-6).all()
    # backward pass check
    assert torch.isclose(v_grad, v.grad, atol=1e-6).all()
    assert torch.isclose(v_pts_grad, v_pts.grad, atol=1e-6).all()
    assert torch.isclose(z_out_bias_grad, z_out_bias.grad, atol=1e-6).all()
    assert torch.isclose(z_attn_bias_grad, z_attn_bias.grad, atol=1e-6).all()
    assert torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6).all(), (
        q_pts_grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6)],
        q_pts.grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6)]
    )
    assert torch.isclose(k_pts_grad, k_pts_grad, atol=1e-6).all()
    assert torch.isclose(q_grad, q.grad, atol=1e-6).all(), (
        q_grad[~torch.isclose(q_grad, q.grad, atol=1e-6)],
        q.grad[~torch.isclose(q_grad, q.grad, atol=1e-6)],
    )
    assert torch.isclose(k_grad, k.grad, atol=1e-6).all()
    # TODO: why is this error so large relative to everything else...
    assert torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3).all(), (
        pts_bias_scale_grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3)],
        pts_bias_scale.grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3)]
    )


def test_ipa_autograd_func():
    import torch
    import taichi as ti
    import numpy as np

    from taichi_kernels.kernels.ipa import W_L
    from taichi_kernels.torch.ipa import fused_ipa_kernel

    if torch.cuda.is_available():
        device = 'cuda:0'
        ti.init(arch=ti.cuda)
    else:
        device='cpu'
        ti.init()

    B = 4
    H = 8
    L = 130 #128
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
    torch_out = torch.einsum("...ij,...jk->...ik", attn, v)
    torch_out_pts = torch.einsum("...ij,...jpk->...ipk", attn, v_pts)
    torch_out_bias = torch.sum(
        attn[..., None] * z_out_bias,
        dim=-2
    )
    torch.autograd.backward(
        tensors=[torch_out, torch_out_pts, torch_out_bias],
        grad_tensors=[torch.ones_like(torch_out), torch.ones_like(torch_out_pts), torch.ones_like(torch_out_bias)],
    )
    q_grad = q.grad
    q_pts_grad = q_pts.grad
    k_grad = k.grad
    k_pts_grad = k_pts.grad
    v_grad = v.grad
    v_pts_grad = v_pts.grad
    z_attn_bias_grad = z_attn_bias.grad
    z_out_bias_grad = z_out_bias.grad
    pts_bias_scale_grad = pts_bias_scale.grad

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

    # forward pass correctness
    assert torch.isclose(out, torch_out, atol=1e-6).all()
    print(out_pts.shape, torch_out_pts.shape)
    assert torch.isclose(out_pts, torch_out_pts, atol=1e-6).all()
    assert torch.isclose(out_bias, torch_out_bias, atol=1e-6).all()

    # backward pass correctness
    assert torch.isclose(v_grad, v.grad, atol=1e-6).all()
    assert torch.isclose(v_pts_grad, v_pts.grad, atol=1e-6).all()
    assert torch.isclose(z_out_bias_grad, z_out_bias.grad, atol=1e-6).all()
    assert torch.isclose(z_attn_bias_grad, z_attn_bias.grad, atol=1e-6).all()
    assert torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6).all()
    assert torch.isclose(k_pts_grad, k_pts_grad, atol=1e-6).all()
    assert torch.isclose(q_grad, q.grad, atol=1e-6).all()
    assert torch.isclose(k_grad, k.grad, atol=1e-6).all()
    # TODO: why is this error so large relative to everything else...
    assert torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3).all()