import numpy as np
import psi4
try:
    from . import params
except SystemError:
    import params


def JK_adv_setup(mints):
    # Building the auxiliary basis
    aux = psi4.core.BasisSet.build(params.mol, fitrole="JKFIT",
                                   other=params.bas_str)

    # Building the zero basis for the 2- and 3-center integrals
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    # Building 3-center integrals
    Qls_tilde = mints.ao_eri(zero_bas, aux, params.bas, params.bas)
    Qls_tilde = np.squeeze(Qls_tilde)

    # Building Coulomb metric
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1.e-14)
    metric = np.squeeze(metric)

    # Building 3-center tensor
    Pls = np.einsum("qls,pq->pls", Qls_tilde, metric)

    return Pls


def make_JK_adv(Pls, g, D, Cocc):
    """
    Function to make the Coulomb and Exchange matrices
    using density-fitting
    """
    # Building the Coulomb matrix ( O(N^2*Naux) )
    chi = np.einsum("pls,ls->p", Pls, D)
    J = np.einsum("pmn,p->mn", Pls, chi)

    # Building the Exchange matrix ( O(N^2*Naux*p) )
    xi1 = np.einsum("qms,sp->qmp", Pls, Cocc)
    xi2 = np.einsum("qnl,lp->qnp", Pls, Cocc)
    K = np.einsum("qmp,qnp->mn", xi1, xi2)

    return J, K
