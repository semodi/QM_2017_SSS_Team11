import pytest
import scf_11
import numpy as np
import psi4

h2o_mole = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")
h2o_mole.update_geometry()
h2o_bas = psi4.core.BasisSet.build(h2o_mole, target="aug-cc-PVDZ")
h2o_nel = 5
h2o = scf_11.Molecule(h2o_mole,h2o_bas,h2o_nel)

def test_scf_basic():
    E_total = scf_11.scf(h2o)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=h2o_mole)
    assert np.allclose(psi4_energy, E_total)

def test_scf_diis():
    E_total = scf_11.scf(h2o, DIIS_mode=True)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=h2o_mole)
    assert np.allclose(psi4_energy, E_total)

def test_scf_jk():
    E_total = scf_11.scf(h2o, JK_mode=True)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=h2o_mole)
    assert np.allclose(psi4_energy, E_total)

def test_scf_jk():
    E_total = scf_11.scf(h2o, JK_mode=True,DIIS_mode=True)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=h2o_mole)
    assert np.allclose(psi4_energy, E_total)

def test_scf_default():
    E_total = scf_11.scf(h2o)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-PVDZ", molecule=h2o_mole)
    assert np.allclose(psi4_energy, E_total)
