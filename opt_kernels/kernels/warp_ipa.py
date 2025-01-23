from typing import Any
import warp as wp
import numpy as np

# TODO: i'm not sure we need to be able to change this
BLOCK_SIZE = 4
PARALLEL_BLOCK_SIZE = 1
SERIAL_BLOCK_SIZE = 32
# BLOCK_SIZE = 2
W_L = float(np.sqrt(1/3))
INF = float(1e3)

# vec3_tensor_type = wp.array(dtype=float, ndim=5)

# TODO: i think there might be some minor cases where
# there are errors to the reference torch implementation
# but these generally only occur at a few positions within a huge tensor
# so for now im ignoring them

def generate_fwd_kernel(
        channel_dim,
        bias_dim,
        n_qk_pts,
        n_v_pts,
        dtype,
    ):
    tensor3d_type = wp.array3d(dtype=dtype)
    tensor4d_type = wp.array4d(dtype=dtype)

    @wp.func
    def square(x: float):
        return wp.pow(x, 2.)

    @wp.kernel(enable_backward=False)
    def ipa_sdpa_fwd(
        q: tensor4d_type,             # [B, H, Q, C]
        q_pts: tensor4d_type,    # [B, H * QK_pts, Q, 3]
        k: tensor4d_type,             # [B, H, KV, C]
        k_pts: tensor4d_type,    # [B, H * QK_pts, KV, 3]
        v: tensor4d_type,             # [B, H, KV, C]
        v_pts: tensor4d_type,    # [B, H * V_pts, KV, 3]
        z_attn_bias: tensor4d_type,   # [B, H, Q, K]
        z_out_bias: tensor4d_type,    # [B, Q, K, C_bias]
        out: tensor4d_type,           # [B, H, Q, C]
        out_pts: tensor4d_type,  # [B, H * V_pts, Q 3]
        out_bias: tensor4d_type,      # [B, H, Q, C_bias]
        L: tensor3d_type,             # [B, H, Q]
        M: tensor3d_type,             # [B, H, Q]
        pts_bias_scale: wp.array(dtype=float),     # [H]
    ):
        # define a bunch of necessary constants
        len_k = k.shape[2]
        num_k_blocks = int(wp.ceil(float(len_k) / float(BLOCK_SIZE)))
        c_qk = q.shape[3]
        w_C = wp.sqrt(2. / (9. * float(n_qk_pts)))
        sdpa_scale = 1. / wp.sqrt(float(c_qk))

        batch, head, block_i = wp.tid()

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

            # add negative sign here
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

        m = wp.tile_load(block_M, block_i, BLOCK_SIZE)
        l = wp.tile_load(block_L, block_i, BLOCK_SIZE)
        _L = m + wp.tile_map(wp.log, l)
        wp.tile_store(block_L, block_i, _L)

    return ipa_sdpa_fwd


def generate_bwd_kernel(
        channel_dim,
        bias_dim,
        n_qk_pts,
        n_v_pts,
        dtype,
    ):
    tensor3d_type = wp.array3d(dtype=dtype)
    tensor4d_type = wp.array4d(dtype=dtype)

    @wp.func
    def square(x: float):
        return wp.pow(x, 2.)

    @wp.kernel(enable_backward=False)
    def ipa_sdpa_bwd(
        q: tensor4d_type,             # [B, H, Q, C]
        q_pts: tensor4d_type,    # [B, H * QK_pts, Q, 3]
        k: tensor4d_type,             # [B, H, KV, C]
        k_pts: tensor4d_type,    # [B, H * QK_pts, KV, 3]
        v: tensor4d_type,             # [B, H, KV, C]
        v_pts: tensor4d_type,    # [B, H * V_pts, KV, 3]
        z_attn_bias: tensor4d_type,   # [B, H, Q, K]
        z_out_bias: tensor4d_type,    # [B, Q, K, C_bias]
        out: tensor4d_type,           # [B, H, Q, C]
        out_pts: tensor4d_type,  # [B, H * V_pts, Q 3]
        out_bias: tensor4d_type,      # [B, H, Q, C_bias]
        L: tensor3d_type,             # [B, H, Q]
        pts_bias_scale: wp.array(dtype=float),     # [H]
        q_grad: tensor4d_type,
        q_pts_grad: tensor4d_type,
        k_grad: tensor4d_type,
        k_pts_grad: tensor4d_type,
        v_grad: tensor4d_type,
        v_pts_grad: tensor4d_type,
        z_attn_bias_grad: tensor4d_type,       # [B, H, Q, K]
        z_out_bias_grad: tensor4d_type,        # [B, H, Q, K, C_bias]
        out_grad: tensor4d_type,
        out_pts_grad: tensor4d_type,
        out_bias_grad: tensor4d_type,
        pts_bias_scale_grad: wp.array3d(dtype=float)  # this needs to be 3d so we can tile_assign
    ):
        # define a bunch of necessary constants
        len_k = k.shape[2]
        num_k_blocks = int(wp.ceil(float(len_k) / float(BLOCK_SIZE)))
        c_qk = q.shape[3]
        w_C = wp.sqrt(2. / (9. * float(n_qk_pts)))
        sdpa_scale = 1. / wp.sqrt(float(c_qk))

        batch, head, block_i = wp.tid()

        block_q = q[batch, head]
        block_k = k[batch, head]
        block_v = v[batch, head]
        block_q_pts = q_pts[batch]
        block_k_pts = k_pts[batch]
        block_v_pts = v_pts[batch]
        block_attn_bias = z_attn_bias[batch, head]
        block_L = L[batch, head]
        block_out = out[batch, head]
        block_out_bias = out_bias[batch, head]
        block_out_pts = out_pts[batch]
        block_out_grad = out_grad[batch, head]
        block_out_bias_grad = out_bias_grad[batch, head]
        block_out_pts_grad = out_pts_grad[batch]

        block_q_grad = q_grad[batch, head]
        block_k_grad = k_grad[batch, head]
        block_v_grad = v_grad[batch, head]
        block_q_pts_grad = q_pts_grad[batch]
        block_k_pts_grad = k_pts_grad[batch]
        block_v_pts_grad = v_pts_grad[batch]
        block_attn_bias_grad = z_attn_bias_grad[batch, head]

        # TODO: parallelize over j rather than i,
        # then we can accumulate into j tiles and save once at the end
        dK_j_tile = wp.tile_zeros(BLOCK_SIZE, channel_dim, dtype=float)
        dV_j_tile = wp.tile_zeros(BLOCK_SIZE, channel_dim, dtype=float)

        for block_j in range(num_k_blocks):
            lse = wp.tile_transpose(wp.tile_load(block_L, block_i, BLOCK_SIZE))
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

            # add negative sign here
            # pt_dist *= pts_bias_scale[head] * w_C / (-2.)

            q_tile = wp.tile_load(block_q, block_i, 0, BLOCK_SIZE, channel_dim)
            k_tile = wp.tile_load(block_k, block_j, 0, BLOCK_SIZE, channel_dim)
            dpa = wp.tile_matmul(q_tile, wp.tile_transpose(k_tile))

            bias = wp.tile_load(block_attn_bias, block_i, block_j, BLOCK_SIZE, BLOCK_SIZE)

            s_ij = W_L * (
                sdpa_scale * dpa
                + bias
                + (pts_bias_scale[head] * w_C / (-2.)) * pt_dist
            )
            p_ij = s_ij + (- wp.tile_broadcast(lse, BLOCK_SIZE, BLOCK_SIZE))
            p_ij = wp.tile_map(wp.exp, p_ij)
            p_ij_T = wp.tile_transpose(p_ij)

            # update dV_j
            out_grad_tile = wp.tile_load(block_out_grad, block_i, 0, BLOCK_SIZE, channel_dim)
            v_grad_tile = wp.tile_matmul(p_ij_T, out_grad_tile)
            # print(v_grad_tile)
            wp.tile_atomic_add(block_v_grad, block_j, 0, v_grad_tile)

            # update dV_pts_j
            for p in range(n_v_pts):
                head_p = head * n_v_pts + p
                out_pts_grad_tile = wp.tile_load(block_out_pts_grad[head_p], block_i, 0, BLOCK_SIZE, 3)
                v_pts_grad_tile = wp.tile_matmul(p_ij_T, out_pts_grad_tile)
                wp.tile_atomic_add(block_v_pts_grad[head_p], block_j, 0, v_pts_grad_tile)

            # update dZ_out_bias_j
            out_bias_grad_tile = wp.tile_load(block_out_bias_grad, block_i, 0, BLOCK_SIZE, bias_dim)
            for i in range(BLOCK_SIZE):
                block_z_out_bias_grad = z_out_bias_grad[batch, block_i * BLOCK_SIZE + i]
                update = wp.tile_matmul(
                    wp.tile_transpose(p_ij[i]),  # j_dim x 1 (i_dim)
                    out_bias_grad_tile[i]  # 1 (i dim) x bias_dim
                )  # j_dim x channel_dim
                wp.tile_atomic_add(block_z_out_bias_grad, block_j, 0, update)

            # compute dP_ij
            dp_ij = wp.tile_zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=float)
            # from `out`
            block_v_tile = wp.tile_load(block_v, block_j, 0, BLOCK_SIZE, channel_dim)
            dp_ij += wp.tile_matmul(out_grad_tile, wp.tile_transpose(block_v_tile))
            # from `out_pts`
            for p in range(n_v_pts):
                head_p = head * n_v_pts + p
                v_pts_tile = wp.tile_load(block_v_pts[head_p], block_j, 0, BLOCK_SIZE, 3)
                out_pts_grad_tile = wp.tile_load(block_out_pts_grad[head_p], block_i, 0, BLOCK_SIZE, 3)
                dp_ij += wp.tile_matmul(out_pts_grad_tile, wp.tile_transpose(v_pts_tile))
            # from `out_bias`
            dp_ij_update = wp.tile_zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=float)
            for i in range(BLOCK_SIZE):
                block_z_out_bias = z_out_bias[batch, block_i * BLOCK_SIZE + i]
                z_out_bias_tile = wp.tile_load(block_z_out_bias, block_j, 0, BLOCK_SIZE, bias_dim)
                update = wp.tile_matmul(
                    out_bias_grad_tile[i],   # 1 (i dim) x bias_dim
                    wp.tile_transpose(z_out_bias_tile)  # bias_dim x j
                )
                wp.tile_assign(dp_ij_update[i], 0, 0, update)
            dp_ij += dp_ij_update


            # compute dS_ij
            # there's a lot of shenanigans going on here
            # for one, since warp doesn't have rowwise operators
            # we're getting around this by doing an outer product and taking the diagonal
            out_tile = wp.tile_load(block_out, block_i, 0, BLOCK_SIZE, channel_dim)
            # from `out`
            d_i_out = wp.tile_matmul(out_grad_tile, wp.tile_transpose(out_tile))

            # from `out_pts`
            d_i_out_pts = wp.tile_zeros(BLOCK_SIZE, BLOCK_SIZE, dtype=float)
            for p in range(n_v_pts):
                head_p = head * n_v_pts + p
                out_pts_tile = wp.tile_load(block_out_pts[head_p], block_i, 0, BLOCK_SIZE, 3)
                out_pts_grad_tile = wp.tile_load(block_out_pts_grad[head_p], block_i, 0, BLOCK_SIZE, 3)
                d_i_out_pts += wp.tile_matmul(out_pts_grad_tile, wp.tile_transpose(out_pts_tile))

            # from `out_bias`
            out_bias_tile = wp.tile_load(block_out_bias, block_i, 0, BLOCK_SIZE, bias_dim)
            d_i_out_bias = wp.tile_matmul(out_bias_grad_tile, wp.tile_transpose(out_bias_tile))

            d_i_out_update = wp.tile_zeros(BLOCK_SIZE, 1, dtype=float)
            d_i_out_pts_update = wp.tile_zeros(BLOCK_SIZE, 1, dtype=float)
            d_i_out_bias_update = wp.tile_zeros(BLOCK_SIZE, 1, dtype=float)
            for i in range(BLOCK_SIZE):
                # in order to take the diagonal element, we have to do this select-transpose-select mess
                # because tile_assign requires a tile as input and we can only select against the outer-most axis
                wp.tile_assign(d_i_out_update[i], 0, 0, wp.tile_transpose(d_i_out[i])[i])
                wp.tile_assign(d_i_out_pts_update[i], 0, 0, wp.tile_transpose(d_i_out_pts[i])[i])
                wp.tile_assign(d_i_out_bias_update[i], 0, 0, wp.tile_transpose(d_i_out_bias[i])[i])
            d_i = d_i_out_update + d_i_out_pts_update + d_i_out_bias_update
            # print(d_i_out_pts)
            # print(d_i_out_pts_update)
            # print(d_i_out_bias)
            # print(d_i_out_bias_update)

            ds_ij = wp.tile_map(
                wp.mul,
                p_ij,
                dp_ij + (- wp.tile_broadcast(d_i, BLOCK_SIZE, BLOCK_SIZE))
            )
            # print(ds_ij)

            # update db_ij
            wp.tile_store(
                block_attn_bias_grad,
                block_i,
                block_j,
                W_L * ds_ij
            )
            # update d_gamma_h (ipa head weights)
            pts_b_ij_grad = (W_L * w_C / (-2.)) * pt_dist
            pts_b_ij_grad = wp.tile_map(
                wp.mul,
                pts_b_ij_grad,
                ds_ij
            )
            wp.tile_atomic_add(
                pts_bias_scale_grad[head],
                0,
                0,
                wp.tile_sum(pts_b_ij_grad)
            )

            scale = W_L * (-pts_bias_scale[head] * w_C / 2.)
            # compute dQ_pts_i
            # compute dK_pts_i
            ds_ij_T = wp.tile_transpose(ds_ij)
            for p in range(n_qk_pts):
                head_p = head * n_qk_pts + p
                q_pts_tile = wp.tile_load(block_q_pts[head_p], block_i, 0, BLOCK_SIZE, 3)
                k_pts_tile = wp.tile_load(block_k_pts[head_p], block_j, 0, BLOCK_SIZE, 3)

                q_pts_grad_q_factor = wp.tile_zeros(BLOCK_SIZE, 3, dtype=float)
                for j in range(BLOCK_SIZE):
                    _scale = wp.tile_broadcast(
                        wp.tile_transpose(ds_ij_T[j]),
                        BLOCK_SIZE,
                        3
                    )
                    q_pts_grad_q_factor += wp.tile_map(wp.mul, q_pts_tile, _scale)
                q_pts_grad_k_factor = wp.tile_matmul(ds_ij, k_pts_tile)
                q_pts_grad_tile = scale * 2. * (q_pts_grad_q_factor + (- q_pts_grad_k_factor))
                wp.tile_atomic_add(block_q_pts_grad[head_p], block_i, 0, q_pts_grad_tile)

                k_pts_grad_q_factor = wp.tile_matmul(ds_ij_T, q_pts_tile)
                k_pts_grad_k_factor = wp.tile_zeros(BLOCK_SIZE, 3, dtype=float)
                for i in range(BLOCK_SIZE):
                    _scale = wp.tile_broadcast(
                        wp.tile_transpose(ds_ij[i]),
                        BLOCK_SIZE,
                        3
                    )
                    k_pts_grad_k_factor += wp.tile_map(wp.mul, k_pts_tile, _scale)
                k_pts_grad_tile = scale * 2. * (k_pts_grad_k_factor + (- k_pts_grad_q_factor))
                wp.tile_atomic_add(block_k_pts_grad[head_p], block_j, 0, k_pts_grad_tile)


            # update dQ_i
            k_tile = wp.tile_load(block_k, block_j, 0, BLOCK_SIZE, channel_dim)
            q_grad_tile = wp.tile_matmul(
                ds_ij,
                k_tile
            ) * W_L * sdpa_scale
            wp.tile_atomic_add(block_q_grad, block_i, 0, q_grad_tile)

            # update dK_j
            q_tile = wp.tile_load(block_q, block_i, 0, BLOCK_SIZE, channel_dim)
            k_grad_tile = wp.tile_matmul(
                ds_ij_T,
                q_tile
            ) * W_L * sdpa_scale
            wp.tile_atomic_add(block_k_grad, block_j, 0, k_grad_tile)

    return ipa_sdpa_bwd


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
    wp.init()

    D = 1
    D_bias = 1
    n_qk_pts = 8
    n_v_pts = 12

    ipa_sdpa_fwd = generate_fwd_kernel(
        channel_dim=D,
        bias_dim=D_bias,
        n_qk_pts=n_qk_pts,
        n_v_pts=n_v_pts,
        dtype=float,
    )
    ipa_sdpa_bwd = generate_bwd_kernel(
        channel_dim=D,
        bias_dim=D_bias,
        n_qk_pts=n_qk_pts,
        n_v_pts=n_v_pts,
        dtype=float,
    )

    B = 2
    H = 8
    L = 16

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
    # print(out)
    print("lse", lse_store)
    print("m_store", m_store)

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
    pts_bias_scale_grad = torch.zeros_like(pts_bias_scale[..., None, None])

    torch_grad_inputs = [
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
    wp.launch_tiled(
        ipa_sdpa_bwd,
        dim=(B, H, math.ceil(L / BLOCK_SIZE)),
        inputs=torch_grad_inputs,
        block_dim=256
    )

    print("native pytorch")

    #(B, H * qk_pts, L, 3) -> (B, H, L, qk_pts, 3)
    q_pts = q_pts.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    k_pts = k_pts.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    v_pts = v_pts.unflatten(1, (H, n_v_pts)).transpose(-2, -3)

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
    s_ij.retain_grad()
    # print(s_ij)
    p_ij = torch.exp(attn - attn.max(dim=-1)[0][..., None])
    tile_l = p_ij.sum(dim=-1)
    torch_lse = attn.max(dim=-1)[0] + tile_l.log()
    print("torch_lse", torch_lse)
    print("torch_max", attn.max(dim=-1)[0])
    attn = torch.softmax(s_ij, dim=-1)
    attn.retain_grad()
    print(attn)
    torch_out = torch.einsum("...ij,...jk->...ik", attn, v)
    torch_out_pts = torch.einsum("...ij,...jpk->...ipk", attn, v_pts)
    torch_out_bias = torch.sum(
        attn[..., None] * z_out_bias[..., None, :, :, :],
        dim=-2
    )
    end = time.time()
    print("time", end-start)
    # print(out_bias.shape, torch_out_bias.shape)
    torch.autograd.backward(
        tensors=[torch_out, torch_out_pts, torch_out_bias],
        grad_tensors=[
            torch.ones_like(torch_out),
            torch.ones_like(torch_out_pts),
            torch.ones_like(torch_out_bias)
        ],
    )
    # print(s_ij.grad)
    # forward pass check
    # print(out[0])
    # print(torch_out[0])
    assert torch.isclose(out, torch_out, atol=1e-6, rtol=1e-4).all()
    out_pts = out_pts.unflatten(1, (H, n_v_pts)).transpose(-2, -3)
    # print(out_pts[0])
    # print(torch_out_pts[0])
    assert torch.isclose(out_pts, torch_out_pts, atol=1e-6, rtol=1e-4).all()
    # print(out_bias[0])
    # print(torch_out_bias[0])
    assert torch.isclose(out_bias, torch_out_bias, atol=1e-6, rtol=1e-4).all()
    # # backward pass check
    print(v_grad[0, 0])
    print(v.grad[0, 0])
    assert torch.isclose(v_grad, v.grad, atol=1e-6).all()
    v_pts_grad = v_pts_grad.unflatten(1, (H, n_v_pts)).transpose(-2, -3)
    assert torch.isclose(v_pts_grad, v_pts.grad, atol=1e-6).all()
    assert torch.isclose(z_out_bias_grad, z_out_bias.grad, atol=1e-6).all()
    assert torch.isclose(z_attn_bias_grad, z_attn_bias.grad, atol=1e-6).all()
    q_pts_grad = q_pts_grad.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    assert torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6).all(), (
        q_pts_grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6)],
        q_pts.grad[~torch.isclose(q_pts_grad, q_pts.grad, atol=1e-6)]
    )
    k_pts_grad = k_pts_grad.unflatten(1, (H, n_qk_pts)).transpose(-2, -3)
    assert torch.isclose(k_pts_grad, k_pts.grad, atol=1e-6).all()
    assert torch.isclose(q_grad, q.grad, atol=1e-6).all(), (
        q_grad[~torch.isclose(q_grad, q.grad, atol=1e-6)],
        q.grad[~torch.isclose(q_grad, q.grad, atol=1e-6)],
    )
    assert torch.isclose(k_grad, k.grad, atol=1e-6).all()
    # # TODO: why is this error so large relative to everything else...
    pts_bias_scale_grad = pts_bias_scale_grad[..., 0, 0]
    assert torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3).all(), (
        pts_bias_scale_grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3)],
        pts_bias_scale.grad[~torch.isclose(pts_bias_scale_grad, pts_bias_scale.grad, atol=1e-3)]
    )
