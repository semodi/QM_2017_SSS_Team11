import numpy as np

def diis(s, r):
    assert len(s) == len(r)

    B = np.zeros([len(s)+1, len(s)+1])

    for i in range(len(s)):
        for j in range(len(s)):
            B[i, j] = np.einsum('ij,ij->',r[i],r[j])
    
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0
    
    rhs = np.zeros((len(B)))
    rhs[-1] = -1
    
    coeff = np.linalg.solve(B, rhs)
    
    F = np.zeros_like(s[0])
    for i in range(len(coeff)-1):
        F += coeff[i]*s[i]
    
    return F
     
