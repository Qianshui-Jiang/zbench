import torch
import numpy as np
f = 1
h = 8
t = 16

def flash_attention(q, k, v, head_scale=1, is_train=False):
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
            qk *= head_scale #  head_scale
            m_cur = np.maximum(np.amax(qk, -1), m_prev)
            l_prev *= np.exp(m_prev - m_cur)

            p = np.exp(qk - m_cur.reshape(-1, 1))
            l_cur = np.sum(p, -1) + l_prev
            l_rcp = 1 / l_cur
            
            s = p * l_rcp.reshape(-1, 1)
            acc *= (l_prev * l_rcp).reshape(-1, 1)
            
            # Below commeneted is for flash attention2
            # s = p
            # acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
   
            acc += np.matmul(s, v_sub)


            m_prev = m_cur
            l_prev = l_cur

        # acc /= l_prev.reshape(-1, 1)  # for flash attention2
        
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev

    if is_train:
        return output, m, l
    else:
        return output

def flash_attention2(q, k, v, head_scale=1, is_train=False):
    output = np.zeros(q.shape, dtype=np.float32)
    m = np.zeros(f, dtype=np.float32)
    l = np.zeros(f, dtype=np.float32)

    block_m = 1
    block_n = 2
    block_head = h
    assert (f % block_m == 0)
    assert (t % block_n == 0)
    """
    for start_m in range(0, f, block_m):  # Q_seq_len loop
        m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
        l_prev = np.zeros([block_m], dtype=np.float32)
        acc = np.zeros([block_m, block_head], dtype=np.float32)
        q_sub = q[start_m: start_m + block_m, :]
        for start_n in range(0, t, block_n): # Q_seq_len loop
            k_sub = k[start_n: start_n+block_n, :]
            v_sub = v[start_n: start_n+block_n, :]
            qk = np.matmul(q_sub, k_sub.T)
            qk *= head_scale #  head_scale
            
            m_cur = np.maximum(np.amax(qk, -1), m_prev)
            l_prev *= np.exp(m_prev - m_cur)

            p = np.exp(qk - m_cur.reshape(-1, 1))          
            l_cur = np.sum(p, -1) + l_prev
            l_rcp = 1 / l_cur
            
            # Below commeneted is for flash attention
            # s = p * l_rcp.reshape(-1, 1)
            # acc *= (l_prev * l_rcp).reshape(-1, 1)

            s = p
            acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
            
            acc += np.matmul(s, v_sub)
   

            m_prev = m_cur
            l_prev = l_cur

        acc /= l_prev.reshape(-1, 1) # final ouput scale of flash attention 2
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev
    """
    for start_m in range(0, f, block_m):  # Q_seq_len loop
        m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
        l_prev = np.zeros([block_m], dtype=np.float32)
        acc = np.zeros([block_m, block_head], dtype=np.float32)
        q_sub = q[start_m: start_m + block_m, :]
        for start_n in range(0, t, block_n): # Q_seq_len loop
            k_sub = k[start_n: start_n+block_n, :]
            v_sub = v[start_n: start_n+block_n, :]
            qk = np.matmul(q_sub, k_sub.T)
            qk *= head_scale #  head_scale
            
            m_cur = np.maximum(np.amax(qk, -1), m_prev)
            p = np.exp(qk - m_cur.reshape(-1, 1)) 
                         

            l_prev *= np.exp(m_prev - m_cur)
            l_prev = np.sum(p, -1) + l_prev
            
            acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
            acc += np.matmul(p, v_sub)
   
            m_prev = m_cur

        acc /= l_prev.reshape(-1, 1) # final ouput scale of flash attention 2
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev

    if is_train:
        return output, m, l
    else:
        return output


def naive_attention(q, k, v, head_scale=1, is_train=False):
    score = np.matmul(q, k.T)
    score *= head_scale #  head_scale
    row_max = np.amax(score, -1).reshape(-1, 1)
    row_sum = np.sum(np.exp(score - row_max), -1).reshape(-1, 1)
    prob = np.exp(score - row_max) / row_sum

    output = np.matmul(prob, v)
    if is_train:
        return output, prob
    else:
        return output


if __name__ == "__main__":


    q = 10* np.random.uniform(-1, 1, size=(f, h)).astype("float16")
    k = 10* np.random.uniform(-1, 1, size=(t, h)).astype("float16")
    v = 10* np.random.uniform(-1, 1, size=(t, h)).astype("float16")


    desired = naive_attention(q, k, v)

    
    flash_att = flash_attention(q, k, v)
    np.testing.assert_allclose(flash_att, desired, rtol=1e-5, atol=1e-5)
    print("----------------[flash_att PASS]----------------")
    
    flash_att2 = flash_attention2(q, k, v)
    np.testing.assert_allclose(flash_att2, desired, rtol=1e-5, atol=1e-5)
    print("----------------[flash_att2 PASS]----------------")
