import scipy as sp
import numpy as np

try:
    from . import molecule
except SystemError:
    import molecule


def soscf(molecule):
    E2.reshape(nel, nel, len(F) - nel, len(F) - nel)
    for i in nel:
        for j in nel:
            for a in (len(F) - nel):
                for b in (len(F) - nel):
                    E2[i, j, a, b] = -molecule.F[a, b] + molecule.F[i, j]
                    E2[i, j, a, b] += 4 * g[i, a, j, b] - g[i, j, a, b] - g[b, j, a, i]

    kappa = scipy.sparse.linalg.cg(E2, molecule.F)

    C = np.dot(molecule.C, scipy.linalg.expm(kappa))

    return C
