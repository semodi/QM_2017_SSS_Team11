# coding: utf-8

import numpy as np
import psi4
try:
    from . import params
    from . import diis
    from . import jk
    from . import molecule
except SystemError:
    import params
    import diis
    import jk
    import molecule

np.set_printoptions(suppress=True, precision=4)

def diag(F, A):
    '''
    Function to diag. a Fock mat. F with orthog. mat. A
    '''

    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def make_JK(Pls, g, D, Cocc, JK_mode):
    '''
    Function to make Coulomb matrix J and Exchange matrix K from
    two-electron integrals g and density D, with flag JK_mode
    '''
    if JK_mode:
        J, K = jk.make_JK_adv(Pls, g, D, Cocc)
    if not JK_mode:
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)
    return J, K


def damp(F_old, F_new, damp_start, damp_value, i):
    '''
    Function for handling convergence damping
    '''

    if i >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new
    return F


def scf(molecule, damp_start=5, damp_value=0.2,
                e_conv=1.e-6, d_conv=1.e-6, JK_mode=False, DIIS_mode=False):
    '''
    Main SCF function, returns HF Energy
    '''
#def scf(molecule, e_conv, d_conv, nel, JK_mode, DIIS_mode, damp_start,
#        damp_value):

    # Use object attributes
    mints = molecule.get_mints()
    mol = molecule.mol
    nel = molecule.nel

    # Constructing kinetic and potential energy arrays
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())

    # Constructing core Hamiltonian
    H = T + V

    # Constructing overlap and electron repulsion integral arrays
    S = np.array(mints.ao_overlap())
    molecule.g = molecule.get_ao_eri()

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    # Constructing initial density matrix
    eps, molecule.C = diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

    # Starting SCF loop
    E_old = 0.0
    F_old = None
    F_list = []
    grad_list = []

    # Setting up basis for JK
    if JK_mode:
        Pls = jk.JK_adv_setup(mints)
    else:
        Pls = 0

    for iteration in range(30):
        # Form J and K
        J, K = make_JK(Pls, molecule.g, D, Cocc, JK_mode)

        # Form new Fock matrix
        F_new = H + 2.0 * J - K

        # Check if DIIS extrapolation is requested
        if DIIS_mode:
            F_list.append(F_new)
            molecule.F = F_new
        else:
            molecule.F = damp(F_old, F_new, damp_start, damp_value, iteration)
            F_old = F_new

        # Build the AO gradient
        grad = A.T @ (F @ D @ S - S @ D @ F) @ A

        # Keep track of gradient root-mean-squared
        grad_rms = np.mean(grad ** 2) ** 0.5

        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + mol.nuclear_repulsion_energy()

        E_diff = E_total - E_old
        E_old = E_total

        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
              (iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (grad_rms < d_conv):
            print("SCF has finished!")
            break

        # Append gradient list if DIIS extrapolation is requested
        if DIIS_mode:
            grad_list.append(grad)
            if iteration > 2:
                molecule.F = diis.diis(F_list, grad_list)

        # Build final density matrix
        if SOSCF_mode:
            molecule.C = soscf(molecule)
        else:
            eps, molecule.C = diag(molecule.F, A)
        Cocc = molecule.C[:, :nel]
        D = Cocc @ Cocc.T

        if (iteration == 29):
            print("SCF steps have reached max.")

#    molecule.C = C
    molecule.D = D
    molecule.eps = eps

    return E_total

if __name__ == "__main__":
    h2o = molecule.Molecule(params.mol,params.bas,params.nel)
    scf(h2o, params.damp_start, params.damp_value, params.e_conv, params.d_conv, params.JK_mode,
        params.DIIS_mode)
