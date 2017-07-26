import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")

e_conv = 1.e-6
d_conv = 1.e-6
nel = 5
damp_value = 0.20
damp_start = 5

