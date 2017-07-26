# coding: utf-8
import input
import psi4

np.set_printoptions(suppress=True, precision=4)

# Built a MintsHelper - helps get integrals from psi4
mints = psi4.core.MintsHelper(input.bas)

nbf = mints.nbf()

if (nbf > 100):
    raise Exception ("More than 100 basis functions!")

