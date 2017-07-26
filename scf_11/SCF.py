# coding: utf-8
from . import params
import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)

# Built a MintsHelper - helps get integrals from psi4
def get_mints(bas):

    mints = psi4.core.MintsHelper(bas)

    nbf = mints.nbf()

    if (nbf > 100):
        raise Exception ("More than 100 basis functions!")

    return mints


def diag(F, A):
    '''
    Function to diag. a Fock mat. F with orthog. mat. A
    '''

    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def make_J(g, D, JK_mode):
    if not JK_mode: 
        J = np.einsum("pqrs,rs->pq", g, D) 
    return J


def make_K(g, D, JK_mode):
    if not JK_mode: 
        K = np.einsum("prqs,rs->pq", g, D) 
    return K


def damp(F_old, F_new, damp_start, damp_value, i):
    if i >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new
    return F

def scf(mints, e_conv, d_conv, nel):
    '''
    Main SCF function
    '''

    # Constructing kinetic and potential energy arrays
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())

    # Constructing core Hamiltonian
    H = T + V

    # Constructing overlap and electron repulsion integral arrays 
    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri()) 

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    # Constructing initial density matrix
    eps, C = diag(H, A)
    Cocc = C[:, :params.nel]
    D = Cocc @ Cocc.T  

    # Starting SCF loop
    E_old = 0.0
    F_old = None
    for iteration in range(30):
        # Form J and K
        J = make_J(g, D, params.JK_mode)
        K = make_K(g, D, params.JK_mode)

        F_new = H + 2.0 * J - K

        F = damp(F_old, F_new, params.damp_start, params.damp_value, iteration)

        F_old = F_new
        
        # Build the AO gradient
        grad = F @ D @ S - S @ D @ F

        grad_rms = np.mean(grad ** 2) ** 0.5

        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + params.mol.nuclear_repulsion_energy()

        E_diff = E_total - E_old
        E_old = E_total
        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
                (iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < params.e_conv) and (grad_rms < params.d_conv):
            print ("SCF has finished!") 
            break

        eps, C = diag(F, A)
        Cocc = C[:, :params.nel]
        D = Cocc @ Cocc.T        
        if (iteration == 29):
            print ("SCF steps have reached max.")
    return E_total

if __name__ == "__main__":
    mints = get_mints(params.bas)

    scf(mints, params.e_conv, params.d_conv, params.nel)
