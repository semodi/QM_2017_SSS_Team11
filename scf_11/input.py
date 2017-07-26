import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")

