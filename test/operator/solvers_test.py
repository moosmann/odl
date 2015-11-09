# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super

# External module imports
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ODL imports
import odl
import odl.operator.solvers as solvers
from odl.util.testutils import all_almost_equal, Timer


class MultiplyOp(odl.Operator):
    """Multiply with a matrix."""

    def __init__(self, matrix, domain=None, range=None):
        dom = odl.Rn(matrix.shape[1]) if domain is None else domain
        ran = odl.Rn(matrix.shape[0]) if range is None else range
        super().__init__(dom, ran, linear=True)
        self.matrix = matrix

    def _apply(self, rhs, out):
        np.dot(self.matrix, rhs.data, out=out.data)

    @property
    def adjoint(self):
        return MultiplyOp(self.matrix.T, self.range, self.domain)


"""Test solutions of the linear equation Ax = b with dense A."""

def test_landweber():
    n = 3

    # Np as validation
    A = np.random.rand(n, n) + np.eye(n) * n
    x = np.random.rand(n)
    # Landweber is slow and needs a decent initial guess
    b = np.dot(A, x)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.zero()
    bvec = rn.element(b)

    # Make operator
    norm = np.linalg.norm(A, ord=2)
    Aop = MultiplyOp(A)

    # Solve using landweber
    solvers.landweber(Aop, xvec, bvec, niter=n*10, omega=1/norm**2)
    
    assert all_almost_equal(x, xvec, places=2)
    assert all_almost_equal(Aop(xvec), b, places=2)

def test_conjugate_gradient():
    n = 3

    # Np as validation
    A = np.random.rand(n, n) + np.eye(n) * n
    x = np.random.rand(n)
    b = np.dot(A, x)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.zero()
    bvec = rn.element(b)

    # Make operator
    Aop = MultiplyOp(A)

    # Solve using conjugate gradient
    solvers.conjugate_gradient_normal(Aop, xvec, bvec, niter=n)
    
    assert all_almost_equal(x, xvec, places=2)
    assert all_almost_equal(Aop(xvec), b, places=2)

def test_gauss_newton():
    n = 10

    # Np as validation
    A = np.random.rand(n, n) + np.eye(n) * n
    x = np.random.rand(n)
    b = np.dot(A, x)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.zero()
    bvec = rn.element(b)

    # Make operator
    Aop = MultiplyOp(A)

    # Solve using conjugate gradient
    solvers.gauss_newton(Aop, xvec, bvec, niter=n*3)
    
    assert all_almost_equal(x, xvec, places=2)
    assert all_almost_equal(Aop(xvec), b, places=2)
    

class ResidualOp(odl.Operator):
    """Calculates op(x) - rhs."""

    def __init__(self, op, rhs):
        super().__init__(op.domain, op.range, linear=False)
        self.op = op
        self.rhs = rhs.copy()

    def _apply(self, x, out):
        self.op(x, out)
        out -= self.rhs

    @property
    def derivative(self, x):
        return self.op.derivative(x)

def test_quasi_newton():
    n = 5

    # Np as validation
    A = np.random.rand(n, n)
    A = np.dot(A.T, A) + np.eye(n) * n

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.zero()
    rhs = rn.element(np.random.rand(n))

    # Make operator
    Aop = MultiplyOp(A)
    Res = ResidualOp(Aop, rhs)

    x_opt = np.linalg.solve(A, rhs)

    # Solve using conjugate gradient
    line_search = solvers.BacktrackingLineSearch(lambda x: x.inner(Aop(x)/2.0 - rhs))
    solvers.quasi_newton(Res, xvec, line_search, niter=10)

    assert all_almost_equal(x_opt, xvec, places=2)
    assert Res(xvec).norm() < 10**-1


class Convolution(odl.Operator):
    def __init__(self, kernel, adjkernel=None):
        if not isinstance(kernel.space, odl.DiscreteL2):
            raise TypeError("Kernel must be a DiscreteL2 vector")

        self.kernel = kernel
        self.adjkernel = (adjkernel if adjkernel is not None
                          else kernel.space.element(kernel[::-1].copy()))
        self.space = kernel.space
        self.norm = float(np.sum(np.abs(self.kernel.ntuple)))
        super().__init__(self.space, self.space, linear=True)

    def _apply(self, rhs, out):
        ndimage.convolve(rhs.ntuple.data, self.kernel.ntuple.data,
                         output=out.ntuple.data, mode='wrap')

    @property
    def adjoint(self):
        return Convolution(self.adjkernel, self.kernel)

    def opnorm(self):
        return self.norm

from odl.operator.solvers import operator_norm, StorePartial

def test_chambolle_pock():

    # Continuous definition of problem
    cont_space = odl.L2(odl.Interval(0, 10))

    # Complicated functions to check performance
    cont_kernel = cont_space.element(lambda x: np.exp(x/2) * np.cos(x*1.172))
    cont_data = cont_space.element(lambda x: x**2 * np.sin(x)**2*(x > 5))

    # Discretization
    discr_space = odl.l2_uniform_discretization(cont_space, 500, impl='numpy')
    kernel = discr_space.element(cont_kernel)
    data = discr_space.element(cont_data)

    # Create operator
    conv = Convolution(kernel)

    opn_partial = StorePartial()
    opn = operator_norm(conv, niter=50, partial=opn_partial)

    # Dampening parameter for landweber
    iterations = 40
    omega = 1/conv.opnorm()**2


    # Display partial
    partial = solvers.ForEachPartial(lambda result: plt.plot(conv(result)[:]))

    opn = 4 * opn
    # assert  conv.opnorm() == opn

    # Test CGN
    plt.figure('CGN')
    plt.plot(data)
    xcgn = discr_space.zero()
    solvers.conjugate_gradient_normal(conv, xcgn, data,
                                       iterations, partial)

    # Chambolle Pock
    plt.figure('Chambolle-Pock')
    plt.plot(data)
    xcp = discr_space.zero()
    solvers.chambolle_pock(conv, xcp, data, iterations,
                           op_norm=opn, partial=partial)

    plt.figure('Chambolle-Pock final')
    plt.plot(data)
    plt.plot(conv(xcp))
    plt.plot(conv(xcgn))


    plt.show()

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\','/')) + ' -v')

