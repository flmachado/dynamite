
import unittest as ut

from itertools import product
from collections import OrderedDict

dnm_args = []

from dynamite import config
config.initialize(dnm_args)

import dynamite.operators as do
from dynamite.tools import build_state,vectonumpy
from dynamite.backend.backend import MSC_dtype
from dynamite.spinconserve import idx_to_state,state_to_idx,subspace_dim
import numpy as np
from math import factorial as fac
from petsc4py.PETSc import Sys
Print = Sys.Print

from helpers import *
import Hamiltonians

class Mapping(ut.TestCase):

    def setUp(self):
        self.L = [7,10]

    def test_dim(self):
        for L in self.L:
            for sz in range(0,L+1):
                with self.subTest(L=L,sz=sz):
                    self.assertEqual(subspace_dim(L,sz),fac(L)//fac(sz)//fac(L-sz))

    def test_map(self):

        # notably, we don't really care what the mapping is, as long as the states
        # map forward and back to the same thing, and the states actually have the
        # right number of up spins

        for L in self.L:
            for sz in range(0,L+1):
                with self.subTest(L=L,sz=sz):
                    idxs = np.arange(subspace_dim(L,sz),dtype=MSC_dtype[0])
                    states = idx_to_state(idxs,L,sz)
                    self.assertTrue(all(bin(x).count('1') == sz for x in states))
                    self.assertTrue(np.all(idxs == state_to_idx(states,L,sz)))

class Values(ut.TestCase):

    def setUp(self):
        config.global_L = 8
        self.sz_tests = [0,1,4]

    def tearDown(self):
        config.global_L = None
        config.global_sz = None

    def test_Hamiltonians(self):
        for name in ['XXYY','XXYYZZ']:
            for sz in self.sz_tests:
                with self.subTest(name=name,sz=sz):
                    config.global_sz = sz
                    d,n = getattr(Hamiltonians,name)(config.global_L)

                    # get rid of the rows outside of the subspace
                    idxs = np.arange(subspace_dim(),dtype=MSC_dtype[0])
                    states = idx_to_state(idxs)
                    states.shape += 1, # make states 2d
                    n = n[states.T,states]

                    r,msg = check_dnm_np(d,n,full_space=False)
                    self.assertTrue(r,msg=msg)

class StateBuilding(ut.TestCase):

    def setUp(self):
        config.global_L = 10
        self.L = config.global_L
        self.sz_tests = [0,1,5,8]

    def tearDown(self):
        config.global_L = None
        config.global_sz = None

    def test_idx(self):
        for sz in self.sz_tests:
            config.global_sz = sz
            for i in [0,
                      # some randomly-picked states
                      int(0.339054706405*subspace_dim()),
                      int(0.933703666179*subspace_dim())]:
                with self.subTest(idx=i):
                    d = build_state(idx=i)
                    n = np.zeros(subspace_dim(),dtype=np.complex128)
                    n[i] = 1

                    r,msg = check_vecs(d,n)
                    self.assertTrue(r,msg=msg)

    def test_state(self):
        tests = [
            ('UUDDUUDDUU',6),
            (int('1101100011',2),6),
        ]
        for state,sz in tests:
            config.global_sz = sz
            with self.subTest(state=state):
                d = build_state(state=state)
                n = np.zeros(subspace_dim(),dtype=np.complex128)

                if isinstance(state,str):
                    s = int(state.replace('U','1').replace('D','0'),2)
                else:
                    s = state

                n[state_to_idx(s)] = 1

                r,msg = check_vecs(d,n)
                self.assertTrue(r,msg=msg)

    def test_random(self):
        for sz in self.sz_tests:
            with self.subTest(sz=sz):
                config.global_sz = sz

                s = build_state(state='random')
                r,msg = check_close(s.norm(),1)
                self.assertTrue(r,msg=msg)

                self.assertTrue(s.size == subspace_dim())

                s = build_state(state='random',seed=0)
                t = build_state(state='random',seed=0)

                r,msg = check_close(s.norm(),1)
                self.assertTrue(r,msg=msg)

                r,msg = check_close(s.dot(t),1)
                self.assertTrue(r,msg=msg)

    def test_buildstate_exceptions(self):
        tests = [
            ('UUDDUUDDUD',6),
            (int('1101100011',2),5)
        ]
        for state,sz in tests:
            with self.subTest(state=state):
                config.global_sz = sz
                with self.assertRaises(ValueError):
                    build_state(state=state)

class Evolve(ut.TestCase):

    def setUp(self):
        config.global_L = 8
        config.global_sz = 4
        self.test_states = [0,
                            int(0.339054706405*subspace_dim()),
                            int(0.933703666179*subspace_dim()),
                            'random']

    def test_Hamiltonians(self):
        for name in Hamiltonians.__all__:
            for idx in self.test_states:
                if idx != 'random':
                    state = int(idx_to_state(idx))
                else:
                    state = idx
                with self.subTest(H=name,state=state):
                    d,n = getattr(Hamiltonians,name)(config.global_L)

                    # get rid of the rows outside of the subspace
                    idxs = np.arange(subspace_dim(),dtype=MSC_dtype[0])
                    states = idx_to_state(idxs)
                    states.shape += 1, # make states 2d
                    n = n[states.T,states]

                    r,msg = check_evolve(d,n,state)
                    self.assertTrue(r,msg=msg)

    def tearDown(self):
        config.spin_conserve = None

class Eigsolve(ut.TestCase):

    def setUp(self):
        config.global_L = 8
        self.sz_tests = [1,4]

    def test_eigsolve(self):

        for sz in self.sz_tests:
            with self.subTest(sz=sz):
                config.global_sz = sz

                d,n = Hamiltonians.XXYYZZ(L=config.global_L)

                # get rid of the rows outside of the subspace
                idxs = np.arange(subspace_dim(),dtype=MSC_dtype[0])
                states = idx_to_state(idxs)
                states.shape += 1, # make states 2d
                n = n[states.T,states]

                with self.subTest(which='smallest'):
                    self.check_eigs(d,n)

                with self.subTest(which='smallest_4'):
                    self.check_eigs(d,n,nev=4)

                with self.subTest(which='exterior'):
                    self.check_eigs(d,n,which='exterior')

                with self.subTest(which='target0'):
                    self.check_eigs(d,n,target=0,nev=2)

                with self.subTest(which='target-1.2'):
                    self.check_eigs(d,n,target=-1.2,nev=2)

    def tearDown(self):
        config.global_sz = None

    def check_eigs(self,d,n,**kwargs):

        evs,evecs = d.eigsolve(getvecs=True,**kwargs)
        nevs,_ = np.linalg.eigh(n)

        if 'nev' in kwargs:
            self.assertGreater(len(evs),kwargs['nev']-1)
        else:
            self.assertGreater(len(evs),0)

        # TODO: should check 'target' option actually gives eigs
        # closest to target

        # make sure every eigenvalue is close to one in the list
        # also check that the eigenvector is correct
        for ev,evec in zip(evs,evecs):
            with self.subTest(ev=ev):
                # there are some matching eigenvalues
                if not np.abs(nevs-ev).min() < 1E-12:
                    pass
                    #print(evs,nevs,n)
                self.assertLess(np.abs(nevs-ev).min(),1E-12)

                # check that the eigenvector is
                # a) an eigenvector and
                # b) has the right eigenvalue
                if ev != 0:
                    err = d.get_mat()*evec / ev - evec
                else:
                    err = d.get_mat()*evec
                errnorm = err.norm(NormType.INFINITY)
                vecnorm = evec.norm(NormType.INFINITY)
                self.assertLess(errnorm,1E-6*vecnorm)

from dynamite.computations import reduced_density_matrix
from dynamite.computations import entanglement_entropy
from petsc4py.PETSc import Vec

# this test uses qutip to test the entanglement entropy computation
# if we don't have qutip, just skip it

try:
    import qutip as qtp
except ImportError:
    qtp = None

@ut.skipIf(qtp is None,'could not find QuTiP')
class Entropy(ut.TestCase):

    def setUp(self):
        config.global_L = 6
        self.cuts = [0,1,3]
        self.shifts = [0,2,4]
        self.sz = [1,3,4]
        self.states = OrderedDict([
            ('product0',{'idx':0}),
            ('product1',{'idx':1}),
            ('random',{'state':'random'})
        ])

    def test_dm_entropy(self):
        for cut in self.cuts:
            for shift in self.shifts:
                for name,state_args in self.states.items():
                    for sz in self.sz:
                        with self.subTest(cut=cut,state=name,sz=sz,shift=shift):
                            config.global_sz = sz
                            state = build_state(**state_args)

                            if cut + shift > config.global_L:
                                with self.assertRaises(ValueError):
                                    ddm = reduced_density_matrix(state,cut,start=shift)
                                with self.assertRaises(ValueError):
                                    dy_EE = entanglement_entropy(state,cut,start=shift)
                                continue
                            else:
                                ddm = reduced_density_matrix(state,cut,start=shift)
                                dy_EE = entanglement_entropy(state,cut,start=shift)

                            np_vec = np.zeros(2**config.global_L,dtype=np.complex128)
                            idxs = idx_to_state(np.arange(state.size))
                            sc_vec = vectonumpy(state)

                            if sc_vec is not None:
                                # only do this on process 0

                                for i,v in zip(idxs,sc_vec):
                                    np_vec[i] = v

                                qtp_state = qtp.Qobj(np_vec,dims=[[2]*config.global_L,
                                                                  [1]*config.global_L])

                                dm = qtp_state * qtp_state.dag()

                                if cut > 0:
                                    dm = dm.ptrace(list(range(config.global_L-cut-shift,
                                                              config.global_L-shift)))
                                    qtp_EE = qtp.entropy_vn(dm)
                                    dm = dm.full()
                                else:
                                    # qutip breaks when you ask it to trace out everything
                                    # maybe I should submit a pull request to them
                                    dm = np.array([[1.+0.0j]])
                                    qtp_EE = 0
                            else:
                                dm = None
                                qtp_EE = None

                            r,msg = check_allclose(dm,ddm)
                            msg += '\nnumpy:\n'+str(dm)+'\ndynamite:\n'+str(ddm)
                            self.assertTrue(r,msg=msg)

                            r,msg = check_close(qtp_EE,dy_EE)
                            self.assertTrue(r,msg=msg)

    def tearDown(self):
        config.global_sz = None

if __name__ == '__main__':

    # # only get output from one process
    # from sys import stderr,argv
    # from os import devnull
    # if PROC_0:
    #     stream = stderr
    # else:
    #     stream = open(devnull,'w')
    # ut.main(testRunner=ut.TextTestRunner(stream=stream))
    from sys import argv
    ut.main(argv=argv)
