import psi4
import numpy as np

class Molecule:

    def __init__(self, mol=None, bas=None):
        self.mol = mol
        self.bas = bas
        self.C = None
        self.D = None
        self.eps = None
        self.ao_eri = None

    def set_geometry(self, geom_str):
        self.mol = psi4.geometry(geom_str)
        self.mol.update_geometry()

    def set_basis(bas_str):
        if(self.mol is None):
            raise Exception('Error: Geometry not defined')
        else:
            self.bas = psi4.core.BasisSet.build(self.mol, target=bas_str)

    def get_mints(self):
        '''
        Built a MintsHelper - helps get integrals from psi4
        '''
        mints = psi4.core.MintsHelper(self.bas)

        nbf = mints.nbf()

        if (nbf > 200):
            raise Exception("More than 200 basis functions!")

        return mints

    def get_ao_eri(self):
        '''
        Returns two electron integrals. If they don't exist
        use mints to calculate and store them.
        '''
        if self.ao_eri == None:
            self.ao_eri = np.array(mints.ao_eri())
        return self.ao_eri
