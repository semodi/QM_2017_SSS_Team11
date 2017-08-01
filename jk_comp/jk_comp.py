import numpy as np
import jk_blas
import psi4

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a ERI tensor
basis = psi4.core.BasisSet.build(mol, target="cc-pVDZ")
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())

# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
J_ref = np.einsum("pqrs,rs->pq", I, D)
K_ref = np.einsum("prqs,rs->pq", I, D)

# Your implementation
# Flatten I into rank-2 tensor

I_flat = I.flatten().reshape(nbf**2,nbf**2)
D_flat = D.flatten()

J = calc_J(I_flat,D_flat) 

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))
