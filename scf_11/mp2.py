import numpy as np

def mp2(molecule, SCS_mode, E_SCF):
    g = molecule.get_ao_eri()
    C = molecule.C
    eps = molecule.eps
    nel = molecule.nel

    # a, b = unoccupied (virtual) MOs
    # i, j = inactive (doubly occupied) MOs
    # Greek letters = general AOs
    
    # Calculate (ia|jb)
    # i = i, j = j, a = a, b = b, mu = m, nu = v, lambda = l, sigma = s
    ivls = np.einsum("mi,mvls->ivls", C, g)
    ivjs = np.einsum("lj,ivls->ivjs", C, ivls)
    iajs = np.einsum("va,ivjs->iajs", C, ivjs)
    iajb = np.einsum("sb,iajs->iajb", C, iajs)

    # Loop over all i, a, j, and b to calculate the energies:
    # nel = number of doubly-occupied orbitals
    # len(C) = total number of orbitals
    # len(C) - nel = number of virtual orbitals
    E_MP2OS = 0.0
    E_MP2SS = 0.0

    for i in range(nel):
        for j in range(nel):
            for a in range(nel, len(C)):
                for b in range(nel, len(C)):
                    E_MP2OS += (iajb[i, a, j, b]*iajb[i, a, j, b])/(eps[i]+eps[j]-eps[a]-eps[b])
                    E_MP2SS += ((iajb[i, a, j, b] - iajb[i, b, j, a])*iajb[i, a, j, b])/(eps[i]+eps[j]-eps[a]-eps[b])


    # Determine the MP2 correction based on whether it should be spin-component scaled (SCS_mode) or not
    if(SCS_mode):
        E_MP2 = (1.0/3.0)*E_MP2SS + (6.0/5.0)*E_MP2OS
    else:
        E_MP2 = E_MP2OS + E_MP2SS

    # Calculate the total energy = E_SCF + E_MP2
    E_tot = E_SCF + E_MP2

    return E_tot
