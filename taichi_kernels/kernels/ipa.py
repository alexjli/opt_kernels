import taichi as ti
import numpy as np

tensor_type = ti.types.ndarray()
vec3_tensor_type = ti.types.ndarray(dtype=ti.types.vector(n=3, dtype=ti.f32))
# TODO: i'm not sure we need to be able to change this
BLOCK_SIZE = 16
W_L = np.sqrt(1/3)
INF = float(1e6)

# TODO: i think there might be some minor cases where
# there are errors to the reference torch implementation
# but these generally only occur at a few positions within a huge tensor
# so for now im ignoring them


@ti.kernel
def ipa_sdpa_fwd(
    q: tensor_type,             # [B, H, Q, C]
    q_pts: vec3_tensor_type,    # [B, H, Q, QK_pts, 3]
    k: tensor_type,             # [B, H, KV, C]
    k_pts: vec3_tensor_type,    # [B, H, KV, QK_pts, 3]
    v: tensor_type,             # [B, H, KV, C]
    v_pts: vec3_tensor_type,    # [B, H, KV, V_pts, 3]
    z_attn_bias: tensor_type,   # [B, H, Q, K]
    z_out_bias: tensor_type,    # [B, H, Q, K, C_bias]
    out: tensor_type,           # [B, H, Q, C]
    out_pts: vec3_tensor_type,  # [B, H, Q, V_pts, 3]
    out_bias: tensor_type,      # [B, H, Q, C_bias]
    L: tensor_type,             # [B, H, Q]
    pts_bias_scale: tensor_type     # [H]
):
    # define a bunch of necessary constants
    n_batch = q.shape[0]
    n_heads = q.shape[1]
    n_qk_pts = q_pts.shape[3]
    n_v_pts = v_pts.shape[3]
    len_q = q.shape[2]
    len_k = k.shape[2]
    num_q_blocks = int(ti.math.ceil(len_q / BLOCK_SIZE))
    num_k_blocks = int(ti.math.ceil(len_k / BLOCK_SIZE))
    c_qk = q.shape[3]
    c_v = v.shape[3]
    c_z = z_out_bias.shape[4]
    w_C = ti.sqrt(2. / (9. * ti.cast(n_qk_pts, ti.f32)))
    sdpa_scale = ti.rsqrt(ti.cast(c_qk, ti.f32))

    for batch, head, block_i in ti.ndrange(n_batch, n_heads, num_q_blocks):
        m = ti.Vector([-INF] * BLOCK_SIZE, ti.f32)
        l = ti.Vector([0] * BLOCK_SIZE, ti.f32)

        ti.loop_config(serialize=True)
        for block_j in ti.ndrange(num_k_blocks):
            curr_m = ti.Vector([-100] * BLOCK_SIZE, ti.f32)
            curr_l = ti.Vector([0] * BLOCK_SIZE, ti.f32)

            s_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
            p_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
            for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
                i_idx = i + block_i * BLOCK_SIZE
                j_idx = j + block_j * BLOCK_SIZE

                if i_idx < len_q and j_idx < len_k:
                    s_ij_bias = 0.
                    for pt_idx in ti.ndrange(n_qk_pts):
                        s_ij_bias += (
                            (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx]) ** 2
                        ).sum()
                    s_ij_bias *= pts_bias_scale[head] * w_C / 2.

                    s_ij_elem = 0.
                    for c in ti.ndrange(c_qk):
                        s_ij_elem += q[batch, head, i_idx, c] * k[batch, head, j_idx, c]
                    s_ij[i, j] = W_L * (
                        sdpa_scale * s_ij_elem
                        + z_attn_bias[batch, head, i_idx, j_idx]
                        - s_ij_bias
                    )

                    ti.atomic_max(curr_m[i], s_ij_elem)
                else:
                    s_ij[i, j] = -INF

            # print(s_ij, block_i, block_j)

            for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
                p_ij_elem = ti.exp(s_ij[i, j] - curr_m[i])
                p_ij[i, j] = p_ij_elem
                curr_l[i] += p_ij_elem

            m_new = ti.max(m, curr_m)
            l_new = ti.exp(m - m_new) * l + ti.exp(curr_m - m_new) * curr_l

            for i in ti.ndrange(BLOCK_SIZE):
                i_idx = i + block_i * BLOCK_SIZE

                if i_idx < len_q:
                    for c in ti.ndrange(c_v):
                        out[batch, head, i_idx, c] *= ti.exp(m[i] - m_new[i]) * l[i]
                    for j, c in ti.ndrange(BLOCK_SIZE, c_v):
                        j_idx = j + block_j * BLOCK_SIZE
                        if j_idx < len_k:
                            out[batch, head, i_idx, c] += ti.exp(curr_m[i] - m_new[i]) * p_ij[i, j] * v[batch, head, j_idx, c]
                    for c in ti.ndrange(c_v):
                        out[batch, head, i_idx, c] /= l_new[i]

                    for pt_idx in ti.ndrange(n_v_pts):
                        out_pts[batch, head, i_idx, pt_idx] *= ti.exp(m[i] - m_new[i]) * l[i]
                    for j, pt_idx in ti.ndrange(BLOCK_SIZE, n_v_pts):
                        j_idx = j + block_j * BLOCK_SIZE
                        if j_idx < len_k:
                            out_pts[batch, head, i_idx, pt_idx] += ti.exp(curr_m[i] - m_new[i]) * p_ij[i, j] * v_pts[batch, head, j_idx, pt_idx]
                    for pt_idx in ti.ndrange(n_v_pts):
                        out_pts[batch, head, i_idx, pt_idx] /= l_new[i]

                    for c in ti.ndrange(c_z):
                        out_bias[batch, head, i_idx, c] *= ti.exp(m[i] - m_new[i]) * l[i]
                    for j, c in ti.ndrange(BLOCK_SIZE, c_z):
                        j_idx = j + block_j * BLOCK_SIZE
                        if j_idx < len_k:
                            out_bias[batch, head, i_idx, c] += ti.exp(curr_m[i] - m_new[i]) * p_ij[i, j] * z_out_bias[batch, head, i_idx, j_idx, c]
                    for c in ti.ndrange(c_z):
                        out_bias[batch, head, i_idx, c] /= l_new[i]

            for i in range(BLOCK_SIZE):
                m[i] = m_new[i]
                l[i] = l_new[i]

        for i in ti.ndrange(BLOCK_SIZE):
            i_idx = i + block_i * BLOCK_SIZE
            if i_idx < len_q:
                L[batch, head, i_idx] = m[i] + ti.log(l[i])

@ti.kernel
def ipa_sdpa_bwd(
    q: tensor_type,                 # [B, H, Q, C]
    q_pts: vec3_tensor_type,        # [B, H, Q, QK_pts, 3]
    k: tensor_type,                 # [B, H, KV, C]
    k_pts: vec3_tensor_type,        # [B, H, KV, QK_pts, 3]
    v: tensor_type,                 # [B, H, KV, C]
    v_pts: vec3_tensor_type,        # [B, H, KV, V_pts, 3]
    z_attn_bias: tensor_type,       # [B, H, Q, K]
    z_out_bias: tensor_type,        # [B, H, Q, K, C_bias]
    out: tensor_type,               # [B, H, Q, C]
    out_pts: vec3_tensor_type,      # [B, H, Q, V_pts, 3]
    out_bias: tensor_type,          # [B, H, Q, C_bias]
    L: tensor_type,                 # [B, H, Q]
    pts_bias_scale: tensor_type,    # [H]
    q_grad: tensor_type,
    q_pts_grad: vec3_tensor_type,
    k_grad: tensor_type,
    k_pts_grad: vec3_tensor_type,
    v_grad: tensor_type,
    v_pts_grad: vec3_tensor_type,
    z_attn_bias_grad: tensor_type,       # [B, H, Q, K]
    z_out_bias_grad: tensor_type,        # [B, H, Q, K, C_bias]
    out_grad: tensor_type,
    out_pts_grad: vec3_tensor_type,
    out_bias_grad: tensor_type,
    pts_bias_scale_grad: tensor_type
):
    # define a bunch of necessary constants
    n_batch = q.shape[0]
    n_heads = q.shape[1]
    n_qk_pts = q_pts.shape[3]
    n_v_pts = v_pts.shape[3]
    len_q = q.shape[2]
    len_k = k.shape[2]
    num_q_blocks = int(ti.math.ceil(len_q / BLOCK_SIZE))
    num_k_blocks = int(ti.math.ceil(len_k / BLOCK_SIZE))
    c_qk = q.shape[3]
    c_v = v.shape[3]
    c_z = z_out_bias.shape[4]
    w_C = ti.sqrt(2. / (9. * ti.cast(n_qk_pts, ti.f32)))
    sdpa_scale = ti.rsqrt(ti.cast(c_qk, ti.f32))

    for batch, head, block_i, block_j in ti.ndrange(n_batch, n_heads, num_q_blocks, num_k_blocks):
        # recompute P_ij
        p_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
        for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE

            if i_idx < len_q:
                s_ij_bias = 0.
                for pt_idx in ti.ndrange(n_qk_pts):
                    s_ij_bias += (
                        (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx]) ** 2
                    ).sum()
                s_ij_bias *= pts_bias_scale[head] * w_C / 2.

                s_ij_elem = 0.
                for c in ti.ndrange(c_qk):
                    s_ij_elem += q[batch, head, i_idx, c] * k[batch, head, j_idx, c]
                s_ij = W_L * (
                    sdpa_scale * s_ij_elem
                    + z_attn_bias[batch, head, i_idx, j_idx]
                    - s_ij_bias
                )
                p_ij[i, j] = ti.exp(s_ij - L[batch, head, i_idx])

        # update dV_j
        for i, j, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_v):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                v_grad[batch, head, j_idx, c] += p_ij[i, j] * out_grad[batch, head, i_idx, c]

        # update dV_pts_j
        for j, pt_idx in ti.ndrange(BLOCK_SIZE, n_v_pts):
            j_idx = j + block_j * BLOCK_SIZE
            v_pts_grad_update = ti.Vector([0] * 3, ti.f32)
            if j_idx < len_k:
                for i in ti.ndrange(BLOCK_SIZE):
                    i_idx = i + block_i * BLOCK_SIZE
                    if i_idx < len_q:
                        v_pts_grad_update += p_ij[i, j] * out_pts_grad[batch, head, i_idx, pt_idx]
                v_pts_grad[batch, head, j_idx, pt_idx] += v_pts_grad_update

        # update dZ_out_bias_j
        for i, j, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_z):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                z_out_bias_grad[batch, head, i_idx, j_idx, c] = p_ij[i, j] * out_bias_grad[batch, head, i_idx, c]

        # compute dP_ij
        dp_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)

        # we need to compute the contribution from `out`, `out_grad`, and `out_bias`
        for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                # from `out`
                for c in ti.ndrange(c_v):
                    dp_ij[i, j] += out_grad[batch, head, i_idx, c] * v[batch, head, j_idx, c]
                # from `out_pts`
                for pt_idx in ti.ndrange(n_v_pts):
                    dp_ij[i, j] += out_pts_grad[batch, head, i_idx, pt_idx].dot(v_pts[batch, head, j_idx, pt_idx])
                # from `out_bias`
                for c in ti.ndrange(c_z):
                    dp_ij[i, j] += out_bias_grad[batch, head, i_idx, c] * z_out_bias[batch, head, i_idx, j_idx, c]

        # compute dS_ij
        ds_ij = ti.Matrix([[0] * BLOCK_SIZE for _ in range(BLOCK_SIZE)], ti.f32)
        for i in ti.ndrange(BLOCK_SIZE):
            i_idx = i + block_i * BLOCK_SIZE
            # we write it this way to prevent unrolling
            # d_i = out_grad[i_idx].dot(out[i_idx])
            d_i = 0.
            if i_idx < len_q:
                for c in ti.ndrange(c_v):
                    d_i += out_grad[batch, head, i_idx, c] * out[batch, head, i_idx, c]
                for pt_idx in ti.ndrange(n_v_pts):
                    d_i += out_pts_grad[batch, head, i_idx, pt_idx].dot(out_pts[batch, head, i_idx, pt_idx])
                for c in ti.ndrange(c_z):
                    d_i += out_bias_grad[batch, head, i_idx, c] * out_bias[batch, head, i_idx, c]
            for j in ti.ndrange(BLOCK_SIZE):
                ds_ij[i, j] = p_ij[i, j] * (dp_ij[i, j] - d_i)

        # update db_ij
        for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                z_attn_bias_grad[batch, head, i_idx, j_idx] += W_L * ds_ij[i, j]

        # update d_gamma_h (ipa head weights)
        for i, j in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                # we write it this way to prevent unrolling
                # s_ij_elem = q[i + block_i * block_size].dot(k[j + block_j * block_size])
                pts_b_ij = 0.
                for pt_idx in range(n_qk_pts):
                    pts_b_ij += ((q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx]) ** 2).sum()
                pts_b_ij *= w_C / 2.
                pts_bias_scale_grad[head] += - W_L * pts_b_ij * ds_ij[i, j]

        # compute dQ_pts_i
        for i, j, pt_idx in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, n_qk_pts):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                scale = W_L * (-pts_bias_scale[head] * w_C / 2.)
                q_pts_grad[batch, head, i_idx, pt_idx] += (
                    scale *
                    2. * (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx])
                    * ds_ij[i, j]
                )

        # compute dK_pts_i
        for j, i, pt_idx in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, n_qk_pts):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                scale = W_L * (-pts_bias_scale[head] * w_C / 2.)
                k_pts_grad[batch, head, j_idx, pt_idx] += (
                    scale *
                    -2. * (q_pts[batch, head, i_idx, pt_idx] - k_pts[batch, head, j_idx, pt_idx])
                    * ds_ij[i, j]
                )

        # update dQ_i
        for i, j, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_qk):
            i_idx = i + block_i * BLOCK_SIZE
            j_idx = j + block_j * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                q_grad[batch, head, i_idx, c] += W_L * sdpa_scale * ds_ij[i, j] * k[batch, head, j_idx, c]

        # update dK_j
        for j, i, c in ti.ndrange(BLOCK_SIZE, BLOCK_SIZE, c_qk):
            j_idx = j + block_j * BLOCK_SIZE
            i_idx = i + block_i * BLOCK_SIZE
            if i_idx < len_q and j_idx < len_k:
                k_grad[batch, head, j_idx, c] += W_L * sdpa_scale * ds_ij[i, j] * q[batch, head, i_idx, c]
