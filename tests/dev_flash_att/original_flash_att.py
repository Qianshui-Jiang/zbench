import torch
import numpy as np

# LLAMA2 test
# f = 1
# t = 16
# h = 8

# SD test
f = 64
t = 80
h = 128 

# f = 16  
# t = 16  
# h = 8   


def test_flash_att_origin():

    torch.manual_seed(456)

    N, d = 16, 8

    Q_mat = torch.rand((N, d))
    K_mat = torch.rand((N, d))
    V_mat = torch.rand((N, d))

    # 执行标准的pytorch softmax和attention计算
    expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
    expected_attention = expected_softmax @ V_mat

    ## 执行safe softmax和attention计算
    # 1st read
    S_mat = Q_mat @ K_mat.T
    row_max = torch.max(S_mat, dim=1).values[:, None]
    # 2nd read
    input_safe = S_mat - row_max
    softmax_numerator = torch.exp(input_safe)
    # 3rd read
    softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]
    # 4th read
    safe_softmax = softmax_numerator / softmax_denominator
    # final matmul (another read / write)
    matmul_result = safe_softmax @ V_mat

    assert torch.allclose(safe_softmax, expected_softmax)
    assert torch.allclose(matmul_result, expected_attention)
    
 

    # 执行标准的pytorch softmax和attention计算
    expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
    expected_attention = expected_softmax @ V_mat


    # 分块（tiling）尺寸，以SRAM的大小计算得到
    Br = 4
    Bc = d

    # flash attention算法流程的第2步，首先在HBM中创建用于存储输出结果的O，全部初始化为0
    O = torch.zeros((N, d))
    # flash attention算法流程的第2步，用来存储softmax的分母值，在HBM中创建
    l = torch.zeros((N, 1))
    # flash attention算法流程的第2步，用来存储每个block的最大值，在HBM中创建
    m = torch.full((N, 1), -torch.inf)

    # 算法流程的第5步，执行外循环 （Tiled loop K/V）
    for block_start_Bc in range(0, N, Bc):
        block_end_Bc = block_start_Bc + Bc
        # line 6, load a block from matmul input tensor
        # 算法流程第6步，从HBM中load Kj, Vj的一个block到SRAM
        Kj = K_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
        Vj = V_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
        # 算法流程第7步，执行内循环 (Tiled loop Q)
        for block_start_Br in range(0, N, Br):
            block_end_Br = block_start_Br + Br
            # 算法流程第8行，从HBM中分别load以下几项到SRAM中
            mi = m[block_start_Br:block_end_Br, :]  # shape Br x 1
            li = l[block_start_Br:block_end_Br, :]  # shape Br x 1
            Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d
            Qi = Q_mat[block_start_Br:block_end_Br, :]  # shape Br x d

            # 算法流程第9行
            Sij = Qi @ Kj.T  # shape Br x Bc

            # 算法流程第10行，计算当前block每行的最大值
            mij_hat = torch.max(Sij, dim=1).values[:, None]  # local max

            # 算法流程第10行，计算softmax的分母
            pij_hat = torch.exp(Sij - mij_hat)
            lij_hat = torch.sum(pij_hat, dim=1)[:, None]  # local sum

            # 算法流程第11行，找到当前block的每行最大值以及之前的最大值
            mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

            # 算法流程第11行，计算softmax的分母，但是带了online计算的校正，此公式与前面说的online safe softmax不一致，
            # 但是是同样的数学表达式，只是从针对标量的逐个计算扩展到了针对逐个向量的计算
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

            # 算法流程第12行，计算每个block的输出值
            Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

            # 算法流程第13行
            m[block_start_Br:block_end_Br, :] = mi_new  # row max
            l[block_start_Br:block_end_Br, :] = li_new  # softmax denominator
            # 算法流程第12行，将Oi再写回到HBM
            O[block_start_Br:block_end_Br, :] = Oi

    assert torch.allclose(O, expected_attention)

def test_flash_att2_origin():

    import torch

    Q_BLOCK_SIZE = 3
    KV_BLOCK_SIZE = 3
    NEG_INF = -1e10  # -infinity
    EPSILON = 1e-10
    Q_LEN = 6
    K_LEN = 6
    Tr = Q_LEN // Q_BLOCK_SIZE
    Tc = K_LEN // KV_BLOCK_SIZE

    Q = torch.randn(1, 1, 6, 4, requires_grad=True).to(device='cpu')
    K = torch.randn(1, 1, 6, 4, requires_grad=True).to(device='cpu')
    V = torch.randn(1, 1, 6, 4, requires_grad=True).to(device='cpu')
    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[..., None]
    m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    # start with Q
    for i in range(Tr):
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]
        
        for j in range(Tc):
            #if j>i: 
            #    continue    # ignore masked      
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]

            S_ij = Qi @ Kj.transpose(2,3)
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            mi_new = torch.maximum(m_block_ij, mi)
            P_ij_hat = torch.exp(S_ij - mi_new)
            l_block_ij = torch.sum(P_ij_hat, dim=-1, keepdims=True) + EPSILON
            li_new = torch.exp(mi - mi_new) * li  + l_block_ij 
            O_i = torch.exp(mi - mi_new) * Oi + P_ij_hat @ Vj
            
            print(f'-----------O{i} = attn( Q{i}, KV[{j}])---------')
            print(O_i)
            
        O_BLOCKS[i] = O_i / li_new # 最后做Scaled
        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new
        
    O = torch.cat(O_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)

def dropout(array, ratio, mask):
    assert (array.shape == mask.shape)
    scale = 1 / (1 - float(ratio))
    array_dp = array * scale
    zero = np.zeros(array.shape, dtype=array.dtype)
    output = np.where(mask, array_dp, zero)
    return output

"""
# def flash_attention(q, k, v, is_train=False):
#     output = np.zeros(q.shape, dtype=np.float32)
#     m = np.zeros(f, dtype=np.float32)
#     l = np.zeros(f, dtype=np.float32)

#     block_m = 2
#     block_n = 2
#     block_head = h
#     assert (f % block_m == 0)
#     assert (t % block_n == 0)
#     for start_m in range(0, f, block_m):
#         m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
#         l_prev = np.zeros([block_m], dtype=np.float32)
#         acc = np.zeros([block_m, block_head], dtype=np.float32)
#         q_sub = q[start_m: start_m + block_m, :]
#         for start_n in range(0, t, block_n):
#             k_sub = k[start_n: start_n+block_n, :]
#             v_sub = v[start_n: start_n+block_n, :]
#             dropout_mask_sub = dropout_mask[start_m: start_m +
#                                             block_m, start_n: start_n+block_n]
#             qk = np.matmul(q_sub, k_sub.T)
#             qk *= head_scale
#             m_cur = np.maximum(np.amax(qk, -1), m_prev)
#             l_prev *= np.exp(m_prev - m_cur)
#             p = np.exp(qk - m_cur.reshape(-1, 1))
#             l_cur = np.sum(p, -1) + l_prev
#             l_rcp = 1 / l_cur
#             s = p * l_rcp.reshape(-1, 1)
#             acc *= (l_prev * l_rcp).reshape(-1, 1)
#             # Below commeneted part is from flash attention2
#             # s = p
#             # acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
#             dp_s = dropout(s, dropout_prob, dropout_mask_sub)
#             acc += np.matmul(dp_s, v_sub)
#             m_prev = m_cur
#             l_prev = l_cur
#         # acc /= l_prev.reshape(-1, 1)
#         output[start_m: start_m+block_m, :] = acc
#         m[start_m: start_m+block_m] = m_prev
#         l[start_m: start_m+block_m] = l_prev

#     if is_train:
#         return output, m, l
#     else:
#         return output
"""

def flash_attention(q, k, v, is_train=False):
    output = np.zeros(q.shape, dtype=np.float32)
    m = np.zeros(f, dtype=np.float32)
    l = np.zeros(f, dtype=np.float32)

    block_m = 1
    block_n = 1
    block_head = h
    assert (f % block_m == 0)
    assert (t % block_n == 0)
    for start_m in range(0, f, block_m):
        m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
        l_prev = np.zeros([block_m], dtype=np.float32)
        acc = np.zeros([block_m, block_head], dtype=np.float32)
        q_sub = q[start_m: start_m + block_m, :]
        for start_n in range(0, t, block_n):
            k_sub = k[start_n: start_n+block_n, :]
            v_sub = v[start_n: start_n+block_n, :]
            qk = np.matmul(q_sub, k_sub.T)
            qk *= 0.001 #  head_scale
            m_cur = np.maximum(np.amax(qk, -1), m_prev)
            l_prev *= np.exp(m_prev - m_cur)

            p = np.exp(qk - m_cur.reshape(-1, 1))
            # print(f"==>> start_n: {start_n}")
            # print(f"==>> m_cur: {m_cur}")
            # print(f"==>> m_cur.shape: {m_cur.shape}")
            
            l_cur = np.sum(p, -1) + l_prev
            l_rcp = 1 / l_cur
            s = p * l_rcp.reshape(-1, 1)
            acc *= (l_prev * l_rcp).reshape(-1, 1)
            # Below commeneted part is from flash attention2
            # s = p
            # acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
   
            acc += np.matmul(s, v_sub)

            # ------------------>>>
            # print(f"==>> s.shape: {s.shape}")
            # print(f"==>> v_sub.shape: {v_sub.shape}")
            m_prev = m_cur
            l_prev = l_cur
        # acc /= l_prev.reshape(-1, 1)
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev

    if is_train:
        return output, m, l
    else:
        return output

def flash_attention2(q, k, v, is_train=False):
    output = np.zeros(q.shape, dtype=np.float32)
    m = np.zeros(f, dtype=np.float32)
    l = np.zeros(f, dtype=np.float32)

    block_m = 1
    block_n = 2
    block_head = h
    assert (f % block_m == 0)
    assert (t % block_n == 0)
    for start_m in range(0, f, block_m):
        m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
        l_prev = np.zeros([block_m], dtype=np.float32)
        acc = np.zeros([block_m, block_head], dtype=np.float32)
        q_sub = q[start_m: start_m + block_m, :]
        for start_n in range(0, t, block_n):
            k_sub = k[start_n: start_n+block_n, :]
            v_sub = v[start_n: start_n+block_n, :]
            qk = np.matmul(q_sub, k_sub.T)
            m_cur = np.maximum(np.amax(qk, -1), m_prev)
            l_prev *= np.exp(m_prev - m_cur)

            p = np.exp(qk - m_cur.reshape(-1, 1))
            # print(f"==>> start_n: {start_n}")
            # print(f"==>> m_cur: {m_cur}")
            # print(f"==>> m_cur.shape: {m_cur.shape}")
            
            l_cur = np.sum(p, -1) + l_prev
            l_rcp = 1 / l_cur
            # Below commeneted part is from flash attention
            # s = p * l_rcp.reshape(-1, 1)
            # acc *= (l_prev * l_rcp).reshape(-1, 1)
            
            # Below commeneted part is from flash attention2
            s = p
            acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
            
            acc += np.matmul(s, v_sub)
   
            # ------------------>>>
            # print(f"==>> s.shape: {s.shape}")
            # print(f"==>> v_sub.shape: {v_sub.shape}")
            m_prev = m_cur
            l_prev = l_cur
        acc /= l_prev.reshape(-1, 1)
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev

    if is_train:
        return output, m, l
    else:
        return output

def dev_flash_decoding(q, k, v, is_train=False):
    output = np.zeros(q.shape, dtype=np.float32)
    m = np.zeros(f, dtype=np.float32)
    l = np.zeros(f, dtype=np.float32)

    block_m = 1
    block_n = 4
    block_head = h
    assert (f % block_m == 0)
    assert (t % block_n == 0)
    for start_m in range(0, f, block_m):
        m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
        l_prev = np.zeros([block_m], dtype=np.float32)
        acc = np.zeros([block_m, block_head], dtype=np.float32)
        q_sub = q[start_m: start_m + block_m, :]
        for start_n in range(0, t, block_n):
            k_sub = k[start_n: start_n+block_n, :]
            v_sub = v[start_n: start_n+block_n, :]
            qk = np.matmul(q_sub, k_sub.T)
            # ---------------------------------------
            m_cur = np.maximum(np.amax(qk, -1), m_prev)
            p = np.exp(qk - m_cur.reshape(-1, 1))

            l_prev *= np.exp(m_prev - m_cur)
            l_cur = np.sum(p, -1) + l_prev
            l_rcp = 1 / l_cur
            # 1. Commeneted below is for flash attention
            s = p * l_rcp.reshape(-1, 1)
            acc *= (l_prev * l_rcp).reshape(-1, 1)

            # 2. Commeneted below is for flash attention2
            # s = p
            # acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
            # ---------------------------------------
            
            acc += np.matmul(s, v_sub)
   
            # ------------------>>>
            # print(f"==>> s.shape: {s.shape}")
            # print(f"==>> v_sub.shape: {v_sub.shape}")
            m_prev = m_cur
            l_prev = l_cur
        # acc /= l_prev.reshape(-1, 1)
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev

    if is_train:
        return output, m, l
    else:
        return output

"""

def naive_attention(q, k, v, is_train=False):
    score = np.matmul(q, k.T)
    score *= head_scale
    row_max = np.amax(score, -1).reshape(-1, 1)
    row_sum = np.sum(np.exp(score - row_max), -1).reshape(-1, 1)
    prob = np.exp(score - row_max) / row_sum
    prob_dp = dropout(prob, dropout_prob, dropout_mask)
    output = np.matmul(prob_dp, v)
    if is_train:
        return output, prob, prob_dp
    else:
        return output

"""


def naive_attention(q, k, v, is_train=False):
    score = np.matmul(q, k.T)
    row_max = np.amax(score, -1).reshape(-1, 1)
    row_sum = np.sum(np.exp(score - row_max), -1).reshape(-1, 1)
    prob = np.exp(score - row_max) / row_sum
    output = np.matmul(prob, v)
    if is_train:
        return output, prob
    else:
        return output

def forward_test(q, k, v):
    desired = naive_attention(q, k, v)
    # actual = flash_attention(q, k, v)
    # actual = flash_attention2(q, k, v)
    actual = dev_flash_decoding(q, k, v)

    # print(f"==>> actual: {actual}")
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)
    print("----------------[PASS]----------------")
    

if __name__ == "__main__":
    # llama2_q_shape = [1, 32, 1, 2048]
    # llama2_k_shape = [1, 32, 128, 2048]
    # llama2_v_shape = [1, 32, 128, 2048]
    
    # f = 1
    # t = 128
    # h = 2048
    
    f = 1
    t = 16
    h = 8

    q = np.random.random(size=(f, h))
    k = np.random.random(size=(t, h))
    v = np.random.random(size=(t, h))
        
    # q = np.ones((f, h))
    # k = np.ones((t, h))
    # v = np.ones((t, h))
    
    # do = np.random.random(size=(f, h))
    # head_scale = 1 / np.sqrt(float(h))
    # dropout_prob = 0.3
    # dropout_mask = np.random.random(size=(f, t)) >= dropout_prob
    
    forward_test(q, k, v)
    # test_flash_att_origin()
    # test_flash_att2_origin()
        