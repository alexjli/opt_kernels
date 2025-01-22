from typing import Any
import warp as wp
import numpy as np

# TODO: i'm not sure we need to be able to change this
BLOCK_SIZE = 8
PARALLEL_BLOCK_SIZE = 1
SERIAL_BLOCK_SIZE = 32
# BLOCK_SIZE = 2
W_L = float(np.sqrt(1/3))
INF = float(1e3)
NUM_HEADS = 2

tensor_type = wp.array(dtype=float, ndim=3)
vec3_tensor_type = wp.array(dtype=wp.vec3, ndim=4)
# vec3_tensor_type = wp.array(dtype=float, ndim=5)

# TODO: i think there might be some minor cases where
# there are errors to the reference torch implementation
# but these generally only occur at a few positions within a huge tensor
# so for now im ignoring them

def generate_fwd_kernel(channel_dim, bias_dim, dtype):
    channel_vector = wp.types.vector(length=channel_dim, dtype=dtype)
    bias_vector = wp.types.vector(length=bias_dim, dtype=dtype)
    channel_tensor_type = wp.array(dtype=float, ndim=4)
    dim5_tensor_type = wp.array(ndim=5, dtype=float)
    bias_tensor_type = wp.array(dtype=bias_vector, ndim=3)
    n_qk_pts = 8
    n_v_pts = 12
    num_heads = NUM_HEADS

    @wp.func
    def square(x: float):
        return wp.pow(x, 2.)

    @wp.kernel
    def ipa_sdpa_fwd(
        q: channel_tensor_type,             # [B, H, Q, C]
        q_pts: channel_tensor_type,    # [B, H * QK_pts, Q, 3]
        k: channel_tensor_type,             # [B, H, KV, C]
        k_pts: channel_tensor_type,    # [B, H * QK_pts, KV, 3]
        v: channel_tensor_type,             # [B, H, KV, C]
        v_pts: channel_tensor_type,    # [B, H * V_pts, KV, 3]
        z_attn_bias: wp.array(dtype=float, ndim=4),   # [B, H, Q, K]
        z_out_bias: channel_tensor_type,    # [B, Q, K, C_bias]
        out: channel_tensor_type,           # [B, H, Q, C]
        out_pts: channel_tensor_type,  # [B, H * V_pts, Q 3]
        out_bias: channel_tensor_type,      # [B, H, Q, C_bias]
        L: wp.array3d(dtype=float),             # [B, H, Q]
        M: wp.array3d(dtype=float),             # [B, H, Q]
        pts_bias_scale: wp.array(dtype=float),     # [H]
        num_k_blocks: int
    ):
        # define a bunch of necessary constants
        # n_qk_pts = q_pts.shape[3]
        # n_v_pts = v_pts.shape[3]
        # len_q = q.shape[2]
        # len_k = k.shape[2]
        # num_k_blocks = int(wp.ceil(float(len_k) / float(BLOCK_SIZE)))
        c_qk = q.shape[3]
        w_C = wp.sqrt(2. / (9. * float(n_qk_pts)))
        sdpa_scale = 1. / wp.sqrt(float(c_qk))

        batch, head, block_i = wp.tid()

        # len_i = wp.min(len_q - block_i * BLOCK_SIZE, BLOCK_SIZE)

        block_q = q[batch, head]
        block_k = k[batch, head]
        block_v = v[batch, head]
        block_q_pts = q_pts[batch]
        block_k_pts = k_pts[batch]
        block_v_pts = v_pts[batch]
        block_attn_bias = z_attn_bias[batch, head]
        block_L = L[batch, head]
        block_M = M[batch, head]
        block_out = out[batch, head]
        block_out_bias = out_bias[batch, head]
        block_out_pts = out_pts[batch]


        for block_j in range(num_k_blocks):
            m = wp.tile_transpose(wp.tile_load(block_M, block_i, BLOCK_SIZE))
            l = wp.tile_transpose(wp.tile_load(block_L, block_i, BLOCK_SIZE))
            out_tile = wp.tile_load(block_out, block_i, 0, BLOCK_SIZE, channel_dim)

            pt_dist = wp.tile_zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=float)

            for p in range(n_qk_pts):
                head_p = head * n_qk_pts + p
                q_pts_tile = wp.tile_load(block_q_pts[head_p], block_i, 0, BLOCK_SIZE, 3)
                k_pts_tile = wp.tile_load(block_k_pts[head_p], block_j, 0, BLOCK_SIZE, 3)
                k_pts_tile_T = wp.tile_transpose(k_pts_tile)
                q_pts_tile_T = wp.tile_transpose(q_pts_tile)

                q_pts_tile_T_sq = wp.tile_map(square, q_pts_tile_T)
                q_pts_norm = wp.tile_transpose(
                    q_pts_tile_T_sq[0] + q_pts_tile_T_sq[1] + q_pts_tile_T_sq[2]
                )
                k_pts_tile_T_sq = wp.tile_map(square, k_pts_tile_T)
                k_pts_norm = k_pts_tile_T_sq[0] + k_pts_tile_T_sq[1] + k_pts_tile_T_sq[2]

                pt_dist_update = (
                    -2.0 * wp.tile_matmul(q_pts_tile, k_pts_tile_T)
                    + wp.tile_broadcast(q_pts_norm, BLOCK_SIZE, BLOCK_SIZE)
                    + wp.tile_broadcast(k_pts_norm, BLOCK_SIZE, BLOCK_SIZE)
                )

                pt_dist += pt_dist_update

            # # add negative sign here
            pt_dist *= pts_bias_scale[head] * w_C / (-2.)

            q_tile = wp.tile_load(block_q, block_i, 0, BLOCK_SIZE, channel_dim)
            k_tile = wp.tile_load(block_k, block_j, 0, BLOCK_SIZE, channel_dim)
            dpa = wp.tile_matmul(q_tile, wp.tile_transpose(k_tile))

            bias = wp.tile_load(block_attn_bias, block_i, block_j, BLOCK_SIZE, BLOCK_SIZE)

            s_ij = W_L * (
                sdpa_scale * dpa
                + bias
                + pt_dist
            )
            # print(s_ij)

            tile_m = wp.tile_ones(BLOCK_SIZE, 1, dtype=float) * (-INF)
            tile_l = wp.tile_zeros(BLOCK_SIZE, 1, dtype=float)

            s_ij_T = wp.tile_transpose(s_ij)
            for j in range(BLOCK_SIZE):
                tile_m = wp.tile_map(
                    wp.max,
                    tile_m,
                    wp.tile_transpose(s_ij_T[j])
                )

            p_ij = s_ij + (- wp.tile_broadcast(tile_m, BLOCK_SIZE, BLOCK_SIZE))
            p_ij = wp.tile_map(wp.exp, p_ij)
            p_ij_T = wp.tile_transpose(p_ij)
            for j in range(BLOCK_SIZE):
                tile_l += wp.tile_transpose(p_ij_T[j])

            # print(tile_l)

            m_new = wp.tile_map(wp.max, m, tile_m)
            l_new = (
                wp.tile_map(
                    wp.mul,
                    wp.tile_map(wp.exp, m + (- m_new)),
                    l
                ) + wp.tile_map(
                    wp.mul,
                    wp.tile_map(wp.exp, tile_m + (- m_new)),
                    tile_l
                )
            )

            old_scale = wp.tile_map(
                wp.mul,
                wp.tile_map(wp.exp, m + (-m_new)),
                l
            )
            # old_scale = wp.tile_broadcast(old_scale, m=BLOCK_SIZE, n=BLOCK_SIZE)
            new_scale = wp.tile_map(wp.exp, tile_m + (-m_new))
            new_scale = wp.tile_broadcast(
                new_scale,
                m=BLOCK_SIZE, n=BLOCK_SIZE
            )
            new_scale = wp.tile_map(wp.mul, new_scale, p_ij)
            channel_div_scale = wp.tile_broadcast(l_new, m=BLOCK_SIZE, n=channel_dim)
            pt_div_scale = wp.tile_broadcast(l_new, m=BLOCK_SIZE, n=3)
            bias_div_scale = wp.tile_broadcast(l_new, m=BLOCK_SIZE, n=bias_dim)

            old_scale_diag = wp.tile_zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=float)
            for i in range(BLOCK_SIZE):
                wp.tile_assign(old_scale_diag, i, i, old_scale[i])

            out_tile = (
                wp.tile_matmul(old_scale_diag, out_tile) +
                wp.tile_matmul(
                    new_scale,
                    wp.tile_load(block_v, block_j, 0, BLOCK_SIZE, channel_dim)
                )
            )
            # print(wp.tile_map(wp.div, new_scale, wp.tile_broadcast(l_new, m=BLOCK_SIZE, n=BLOCK_SIZE)))
            out_tile = wp.tile_map(wp.div, out_tile, channel_div_scale)

            for p in range(n_v_pts):
                head_p = head * n_v_pts + p
                out_pts_tile = wp.tile_load(block_out_pts[head_p], block_i, 0, BLOCK_SIZE, 3)
                out_pts_tile = (
                    wp.tile_matmul(old_scale_diag, out_pts_tile) +
                    wp.tile_matmul(
                        new_scale,
                        wp.tile_load(block_v_pts[head_p], block_j, 0, BLOCK_SIZE, 3)
                    )
                )
                out_pts_tile = wp.tile_map(wp.div, out_pts_tile, pt_div_scale)
                wp.tile_store(block_out_pts[head_p], block_i, 0, out_pts_tile)

            out_bias_tile = wp.tile_load(block_out_bias, block_i, 0, BLOCK_SIZE, bias_dim)
            update_tile = wp.tile_zeros(BLOCK_SIZE, bias_dim, dtype=float)
            for i in range(BLOCK_SIZE):
                block_z_out_bias = z_out_bias[batch, block_i * BLOCK_SIZE + i]
                update = wp.tile_load(block_z_out_bias, block_j, 0, BLOCK_SIZE, bias_dim)
                update = wp.tile_matmul(
                    new_scale[i],
                    update
                )
                wp.tile_assign(update_tile[i], 0, 0, update)
            out_bias_tile = wp.tile_matmul(old_scale_diag, out_bias_tile) + update_tile
            out_bias_tile = wp.tile_map(wp.div, out_bias_tile, bias_div_scale)

            wp.tile_store(block_L, block_i, wp.tile_transpose(l_new))
            wp.tile_store(block_M, block_i, wp.tile_transpose(m_new))
            wp.tile_store(block_out, block_i, 0, out_tile)
            wp.tile_store(block_out_bias, block_i, 0, out_bias_tile)

        m = wp.tile_transpose(wp.tile_load(block_M, block_i, BLOCK_SIZE))
        l = wp.tile_transpose(wp.tile_load(block_L, block_i, BLOCK_SIZE))
        _L = m + wp.tile_map(wp.log, l)
        wp.tile_store(block_L, block_i, _L)
        # wp.tile_store(out_pts[batch, head], block_i, out_pts_tile)
        # wp.tile_store(out_bias[batch, head], block_i, 0, out_bias_tile)

    return ipa_sdpa_fwd


# @ti.kernel
# def ipa_sdpa_bwd(
#     q: tensor_type,                 # [B, H, Q, C]
#     q_pts: vec3_tensor_type,        # [B, H, Q, QK_pts, 3]
#     k: tensor_type,                 # [B, H, KV, C]
#     k_pts: vec3_tensor_type,        # [B, H, KV, QK_pts, 3]
#     v: tensor_type,                 # [B, H, KV, C]
#     v_pts: vec3_tensor_type,        # [B, H, KV, V_pts, 3]
#     z_attn_bias: tensor_type,       # [B, H, Q, K]
#     z_out_bias: tensor_type,        # [B, H, Q, K, C_bias]
#     out: tensor_type,               # [B, H, Q, C]
#     out_pts: vec3_tensor_type,      # [B, H, Q, V_pts, 3]
#     out_bias: tensor_type,          # [B, H, Q, C_bias]
#     L: tensor_type,                 # [B, H, Q]
#     pts_bias_scale: tensor_type,    # [H]
#     q_grad: tensor_type,
#     q_pts_grad: vec3_tensor_type,
#     k_grad: tensor_type,
#     k_pts_grad: vec3_tensor_type,
#     v_grad: tensor_type,
#     v_pts_grad: vec3_tensor_type,
#     z_attn_bias_grad: tensor_type,       # [B, H, Q, K]
#     z_out_bias_grad: tensor_type,        # [B, H, Q, K, C_bias]
#     out_grad: tensor_type,
#     out_pts_grad: vec3_tensor_type,
#     out_bias_grad: tensor_type,
#     pts_bias_scale_grad: tensor_type
# ):
#     # define a bunch of necessary constants
#     n_batch = q.shape[0]
#     n_heads = q.shape[1]
#     n_qk_pts = q_pts.shape[3]
#     n_v_pts = v_pts.shape[3]
#     len_q = q.shape[2]
#     len_k = k.shape[2]
#     num_q_blocks = int(ti.math.ceil(len_q / BLOCK_SIZE))
#     num_k_blocks = int(ti.math.ceil(len_k / BLOCK_SIZE))
#     c_qk = q.shape[3]
#     c_v = v.shape[3]
#     c_z = z_out_bias.shape[3]
#     w_C = ti.sqrt(2. / (9. * ti.cast(n_qk_pts, ti.f32)))
#     sdpa_scale = ti.rsqrt(ti.cast(c_qk, ti.f32))
#
#     for batch, head, block_i, block_j in ti.ndrange(n_batch, n_heads, num_q_blocks, num_k_blocks):
#         # recompute P_ij
#         p_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
#         for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#
#             if i_idx < len_q:
#                 s_ij_bias = 0.
#                 for pt_idx in ti.ndrange(n_qk_pts):
#                     s_ij_bias += (
#                         (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx]) ** 2
#                     ).sum()
#                 s_ij_bias *= pts_bias_scale[head] * w_C / 2.
#
#                 s_ij_elem = 0.
#                 for c in ti.ndrange(c_qk):
#                     s_ij_elem += q[batch, head, i_idx, c] * k[batch, head, j_idx, c]
#                 s_ij = W_L * (
#                     sdpa_scale * s_ij_elem
#                     + z_attn_bias[batch, head, i_idx, j_idx]
#                     - s_ij_bias
#                 )
#                 p_ij[i, j] = ti.exp(s_ij - L[batch, head, i_idx])
#
#         # update dV_j
#         for i, j, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_v):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 v_grad[batch, head, j_idx, c] += p_ij[i, j] * out_grad[batch, head, i_idx, c]
#
#         # update dV_pts_j
#         for j, pt_idx in ti.ndrange(BLOCK_SIZE, n_v_pts):
#             j_idx = j + block_j * BLOCK_SIZE
#             v_pts_grad_update = ti.Vector([0] * 3, ti.f32)
#             if j_idx < len_k:
#                 for i in ti.ndrange(BLOCK_SIZE):
#                     i_idx = i + block_i * BLOCK_SIZE
#                     if i_idx < len_q:
#                         v_pts_grad_update += p_ij[i, j] * out_pts_grad[batch, head, i_idx, pt_idx]
#                 v_pts_grad[batch, head, j_idx, pt_idx] += v_pts_grad_update
#
#         # update dZ_out_bias_j
#         for i, j, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_z):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 z_out_bias_grad[batch, i_idx, j_idx, c] += p_ij[i, j] * out_bias_grad[batch, head, i_idx, c]
#
#         # compute dP_ij
#         dp_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
#
#         # we need to compute the contribution from `out`, `out_grad`, and `out_bias`
#         for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 # from `out`
#                 for c in ti.ndrange(c_v):
#                     dp_ij[i, j] += out_grad[batch, head, i_idx, c] * v[batch, head, j_idx, c]
#                 # from `out_pts`
#                 for pt_idx in ti.ndrange(n_v_pts):
#                     dp_ij[i, j] += out_pts_grad[batch, head, i_idx, pt_idx].dot(v_pts[batch, head, j_idx, pt_idx])
#                 # from `out_bias`
#                 for c in ti.ndrange(c_z):
#                     dp_ij[i, j] += out_bias_grad[batch, head, i_idx, c] * z_out_bias[batch, i_idx, j_idx, c]
#
#         # compute dS_ij
#         ds_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
#         for i in ti.ndrange(BLOCK_SIZE):
#             i_idx = i + block_i * BLOCK_SIZE
#             # we write it this way to prevent unrolling
#             # d_i = out_grad[i_idx].dot(out[i_idx])
#             d_i = 0.
#             if i_idx < len_q:
#                 for c in ti.ndrange(c_v):
#                     d_i += out_grad[batch, head, i_idx, c] * out[batch, head, i_idx, c]
#                 for pt_idx in ti.ndrange(n_v_pts):
#                     d_i += out_pts_grad[batch, head, i_idx, pt_idx].dot(out_pts[batch, head, i_idx, pt_idx])
#                 for c in ti.ndrange(c_z):
#                     d_i += out_bias_grad[batch, head, i_idx, c] * out_bias[batch, head, i_idx, c]
#             for j in ti.ndrange(BLOCK_SIZE):
#                 ds_ij[i, j] = p_ij[i, j] * (dp_ij[i, j] - d_i)
#
#         # update db_ij
#         for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 z_attn_bias_grad[batch, head, i_idx, j_idx] += W_L * ds_ij[i, j]
#
#         # update d_gamma_h (ipa head weights)
#         for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 # we write it this way to prevent unrolling
#                 # s_ij_elem = q[i + block_i * block_size].dot(k[j + block_j * block_size])
#                 pts_b_ij = 0.
#                 for pt_idx in range(n_qk_pts):
#                     pts_b_ij += ((q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx]) ** 2).sum()
#                 pts_b_ij *= w_C / 2.
#                 pts_bias_scale_grad[head] += - W_L * pts_b_ij * ds_ij[i, j]
#
#         # compute dQ_pts_i
#         for i, j, pt_idx in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, n_qk_pts):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 scale = W_L * (-pts_bias_scale[head] * w_C / 2.)
#                 q_pts_grad[batch, head, i_idx, pt_idx] += (
#                     scale *
#                     2. * (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx])
#                     * ds_ij[i, j]
#                 )
#
#         # compute dK_pts_i
#         for j, i, pt_idx in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, n_qk_pts):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 scale = W_L * (-pts_bias_scale[head] * w_C / 2.)
#                 k_pts_grad[batch, head, j_idx, pt_idx] += (
#                     scale *
#                     -2. * (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx])
#                     * ds_ij[i, j]
#                 )
#
#         # update dQ_i
#         for i, j, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_qk):
#             i_idx = i + block_i * BLOCK_SIZE
#             j_idx = j + block_j * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 q_grad[batch, head, i_idx, c] += W_L * sdpa_scale * ds_ij[i, j] * k[batch, head, j_idx, c]
#
#         # update dK_j
#         for j, i, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_qk):
#             j_idx = j + block_j * BLOCK_SIZE
#             i_idx = i + block_i * BLOCK_SIZE
#             if i_idx < len_q and j_idx < len_k:
#                 k_grad[batch, head, j_idx, c] += W_L * sdpa_scale * ds_ij[i, j] * q[batch, head, i_idx, c]


if __name__ == '__main__':
    import torch
    import warp as wp
    import numpy as np
    import math
    import time

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device='cpu'
    wp.config.enable_backward = False
    wp.init()

    B = 1
    H = NUM_HEADS
    L = 64
    D = 32
    D_bias = 16
    n_qk_pts = 8
    n_v_pts = 12

    ipa_sdpa_fwd = generate_fwd_kernel(channel_dim=D, bias_dim=D_bias, dtype=float)

    q = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    k = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    v = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    q_pts = torch.randn((B, H * n_qk_pts, L, 3), dtype=torch.float32, device=device)
    k_pts = torch.randn((B, H * n_qk_pts, L, 3), dtype=torch.float32, device=device)
    v_pts = torch.randn((B, H * n_v_pts, L, 3), dtype=torch.float32, device=device)
    z_attn_bias = torch.randn((B, H, L, L), dtype=torch.float32, device=device)
    z_out_bias = torch.randn((B, L, L, D_bias), dtype=torch.float32, device=device)
    pts_bias_scale = torch.ones(H, dtype=torch.float32, device=device)

    out = torch.zeros_like(q)
    out_pts = torch.zeros_like(v_pts)
    out_bias = torch.zeros((B, H, L, D_bias), dtype=torch.float32, device=device)

    lse_store = torch.zeros((B, H, L), dtype=torch.float32, device=device)
    m_store = torch.full((B, H, L), fill_value=-INF, dtype=torch.float32, device=device)

    torch_inputs = [
            q, #.transpose(-1, -2),
            q_pts,
            k, #.transpose(-1, -2),
            k_pts,
            v, #.transpose(-1, -2),
            v_pts,
            z_attn_bias,
            z_out_bias,
            out,
            out_pts,
            out_bias,
            lse_store,
            m_store,
            pts_bias_scale,
            math.ceil(L / BLOCK_SIZE)
    ]
    # wp_inputs = []
    # for idx, t in enumerate(torch_inputs):
    #     print(idx)
    #     wp_inputs.append(wp.from_torch(t, return_ctype=True))

    print("warp kernel")
    wp.launch_tiled(
        ipa_sdpa_fwd,
        dim=(B, H, math.ceil(L / BLOCK_SIZE)),
        inputs=torch_inputs,
        block_dim=256
    )

    L = 256
    D = 128

    q = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    k = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    v = torch.randn((B, H, L, D), dtype=torch.float32, device=device)
    q_pts = torch.randn((B, H * n_qk_pts, L, 3), dtype=torch.float32, device=device)
    k_pts = torch.randn((B, H * n_qk_pts, L, 3), dtype=torch.float32, device=device)
    v_pts = torch.randn((B, H * n_v_pts, L, 3), dtype=torch.float32, device=device)
    z_attn_bias = torch.randn((B, H, L, L), dtype=torch.float32, device=device)
    z_out_bias = torch.randn((B, L, L, D_bias), dtype=torch.float32, device=device)
    pts_bias_scale = torch.ones(H, dtype=torch.float32, device=device)

    out = torch.zeros_like(q)
    out_pts = torch.zeros_like(v_pts)
    out_bias = torch.zeros((B, H, L, D_bias), dtype=torch.float32, device=device)

    lse_store = torch.zeros((B, H, L), dtype=torch.float32, device=device)
    m_store = torch.full((B, H, L), fill_value=-INF, dtype=torch.float32, device=device)

    torch_inputs = [
            q, #.transpose(-1, -2),
            q_pts,
            k, #.transpose(-1, -2),
            k_pts,
            v, #.transpose(-1, -2),
            v_pts,
            z_attn_bias,
            z_out_bias,
            out,
            out_pts,
            out_bias,
            lse_store,
            m_store,
            pts_bias_scale,
            math.ceil(L / BLOCK_SIZE)
    ]
    start = time.time()
    wp.launch_tiled(
        ipa_sdpa_fwd,
        dim=(B, H, math.ceil(L / BLOCK_SIZE)),
        inputs=torch_inputs,
        block_dim=256
    )
    end = time.time()
    print("time", end-start)
    # print(out)
    # print(lse_store)

    # q_grad = torch.zeros_like(q)
    # k_grad = torch.zeros_like(k)
    # v_grad = torch.zeros_like(v)
    # q_pts_grad = torch.zeros_like(q_pts)
    # k_pts_grad = torch.zeros_like(k_pts)
    # v_pts_grad = torch.zeros_like(v_pts)
    # z_attn_bias_grad = torch.zeros_like(z_attn_bias)
    # z_out_bias_grad = torch.zeros_like(z_out_bias)
    # out_grad = torch.ones_like(out)
    # out_pts_grad = torch.ones_like(out_pts)
    # out_bias_grad = torch.ones_like(out_bias)
    # pts_bias_scale_grad = torch.zeros_like(pts_bias_scale)

    # ipa_sdpa_bwd(
    #     q=q,
    #     q_pts=q_pts,
    #     k=k,
    #     k_pts=k_pts,
    #     v=v,
    #     v_pts=v_pts,
    #     z_attn_bias=z_attn_bias,
    #     z_out_bias=z_out_bias,
    #     out=out,
    #     out_pts=out_pts,
    #     out_bias=out_bias,
    #     L=lse_store,
    #     pts_bias_scale=pts_bias_scale,
    #     q_grad=q_grad,
    #     q_pts_grad=q_pts_grad,
    #     k_grad=k_grad,
    #     k_pts_grad=k_pts_grad,
    #     v_grad=v_grad,
    #     v_pts_grad=v_pts_grad,
    #     z_attn_bias_grad=z_attn_bias_grad,
    #     z_out_bias_grad=z_out_bias_grad,
    #     out_grad=out_grad,
    #     out_pts_grad=out_pts_grad,
    #     out_bias_grad=out_bias_grad,
    #     pts_bias_scale_grad=pts_bias_scale_grad
    # )

    # for t in [
    #     q, q_pts,
    #     k, k_pts,
    #     v, v_pts,
    #     z_attn_bias,
    #     z_out_bias,
    #     pts_bias_scale
    # ]:
    #     t.requires_grad = True

    print("native pytorch")

    #(B, H * qk_pts, L, 3) -> (B, H, L, qk_pts, 3)
    q_pts = q_pts.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    k_pts = k_pts.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    v_pts = v_pts.unflatten(1, (H, n_v_pts)).transpose(-2, -3)

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
    attn = attn * W_L
    s_ij = attn
    # print(s_ij)
    p_ij = torch.exp(attn - attn.max(dim=-1)[0][..., None])
    tile_l = p_ij.sum(dim=-1)
    attn = torch.softmax(attn, dim=-1)
    torch_out = torch.einsum("...ij,...jk->...ik", attn, v)
    torch_out_pts = torch.einsum("...ij,...jpk->...ipk", attn, v_pts)
    torch_out_bias = torch.sum(
        attn[..., None] * z_out_bias[..., None, :, :, :],
        dim=-2
    )
    end = time.time()
    print("time", end-start)
    # print(out_bias.shape, torch_out_bias.shape)
    # torch.autograd.backward(
    #     tensors=[torch_out, torch_out_pts, torch_out_bias],
    #     grad_tensors=[out_grad, out_pts_grad, out_bias_grad],
    # )
    # forward pass check
    # print(out[0])
    # print(torch_out[0])
    assert torch.isclose(out, torch_out, atol=1e-6).all()
    out_pts = out_pts.unflatten(1, (H, n_v_pts)).transpose(-2, -3)
    print(out_pts[0])
    print(torch_out_pts[0])
    assert torch.isclose(out_pts, torch_out_pts, atol=1e-6).all()
    # print(out_bias[0])
    # print(torch_out_bias[0])
    assert torch.isclose(out_bias, torch_out_bias, atol=1e-6).all()
    # # backward pass check
    # assert torch.isclose(v_grad, v.grad, atol=1e-6).all()
    # assert torch.isclose(v_pts_grad, v_pts.grad, atol=1e-6).all()
    # assert torch.isclose(z_out_bias_grad, z_out_bias.grad, atol=1e-6).all()
    # assert torch.isclose(z_attn_bias_grad, z_attn_bias.grad, atol=1e-6).all()
    # assert torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6).all(), (
    #     q_pts_grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6)],
    #     q_pts.grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6)]
    # )
    # assert torch.isclose(k_pts_grad, k_pts_grad, atol=1e-6).all()
    # assert torch.isclose(q_grad, q.grad, atol=1e-6).all(), (
    #     q_grad[~torch.isclose(q_grad, q.grad, atol=1e-6)],
    #     q.grad[~torch.isclose(q_grad, q.grad, atol=1e-6)],
    # )
    # assert torch.isclose(k_grad, k.grad, atol=1e-6).all()
    # # TODO: why is this error so large relative to everything else...
    # assert torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3).all(), (
    #     pts_bias_scale_grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3)],
    #     pts_bias_scale.grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3)]
    # )
