import torch 
torch.manual_seed(123)

if __name__ == "__main__":
    # # Softmax
    # A = torch.randn(2, 6)
    # A_exp = torch.exp(A)
    # A_sum = torch.sum(A_exp, dim=1).unsqueeze(1)
    # print(A_sum)
    # print("A_sum.shape():", A_sum.shape)
    # P = A_exp / A_sum # Broadcast
    # print(A)
    # print("A.shape():", A.shape)
    # print(P)
    # print("P.shape():", P.shape)
    
    # # online SoftMax 2-pass
    # N = 6
    # m = torch.tensor(-1000.0)
    # d = 0
    # b = torch.randn(N)
    # a = torch.zeros(N)
    # print(a)

    # for i in range(N):
    #     m_pre = m
    #     m = torch.max(m, a[i])
    #     d = d * (m_pre - m).exp() + (a[i] - m).exp()
        
    # for i in range(N):
    #     b[i] = (a[i]-m).exp() / d
    # print(b)
    # print(torch.sum(b))

    # flast attention
    import torch

    NEG_INF = -1e10  # -infinity
    EPSILON = 1e-10

    Q_LEN = 6
    K_LEN = 6
    Q_BLOCK_SIZE = 3 # 
    KV_BLOCK_SIZE = 3
    Tr = Q_LEN // Q_BLOCK_SIZE
    Tc = K_LEN // KV_BLOCK_SIZE

    Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
    K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
    V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[..., None]
    m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    for j in range(Tr):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]
        for i in range(Tc):
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)
            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_ij = torch.exp(S_ij - m_block_ij)
            l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)
            
            mi_new = torch.maximum(m_block_ij, mi)
            
            li_new = torch.exp(mi - mi_new) * li  \
                + torch.exp(m_block_ij - mi_new) * l_block_ij 

            O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                        +(torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            print(f'-----------Attn : Q{i}xK{j}---------')
    #         print(O_BLOCKS[i].shape)
            print(O_BLOCKS[0])
            print(O_BLOCKS[1])
            print('\n')
            
            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new

    O = torch.cat(O_BLOCKS, dim=2)
    l = torch.cat(l_BLOCKS, dim=2)
    m = torch.cat(m_BLOCKS, dim=2)