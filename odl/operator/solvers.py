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

"""General and optimized equation system solvers in linear spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import next, object, range, super
import numpy as np

# ODL imports
from odl.set.pspace import ProductSpace
from odl.operator.operator import OperatorComp, OperatorSum, Operator
from odl.operator.default_ops import IdentityOperator

# pylint: disable=invalid-name, too-many-arguments
# pylint: disable=abstract-method
#pylint: disable=E1004

class StorePartial(object):
    """ Simple object for storing all partial results of the solvers
    """
    def __init__(self):
        self.results = []

    def send(self, result):
        """ append result to results list
        """
        try:
            self.results.append(result.copy())
        except AttributeError:
            self.results.append(result)

    def __iter__(self):
        return self.results.__iter__()


class ForEachPartial(object):
    """ Simple object for applying a function to each iterate
    """
    def __init__(self, function):
        self.function = function

    def send(self, result):
        """ Applies function to result
        """
        self.function(result)


class PrintIterationPartial(object):
    """ Prints the interation count
    """
    def __init__(self):
        self.iter = 0

    def send(self, _):
        """ Print the current iteration
        """
        print("iter = {}".format(self.iter))
        self.iter += 1


class PrintStatusPartial(object):
    """ Prints the interation count and current norm of each iterate
    """
    def __init__(self):
        self.iter = 0

    def send(self, result):
        """ Print the current iteration and norm
        """
        print("iter = {}, norm = {}".format(self.iter, result.norm()))
        self.iter += 1


def landweber(op, x, rhs, niter=1, omega=1, partial=None):
    """ General and efficient implementation of Landweber iteration

    x <- x - omega * (A')^* (Ax - rhs)
    """

    # Reusable temporaries
    tmp_ran = op.range.element()
    tmp_dom = op.domain.element()

    for _ in range(niter):
        op(x, out=tmp_ran)
        tmp_ran -= rhs
        op.derivative(x).adjoint(tmp_ran, out=tmp_dom)
        x.lincomb(1, x, -omega, tmp_dom)

        if partial is not None:
            partial.send(x)


def conjugate_gradient(op, x, rhs, niter=1, partial=None):
    """ Optimized version of CGN, uses no temporaries etc.
    """
    if op.domain != op.range:
        raise TypeError('Operator needs to be self adjoint')

    r = op(x)
    r.lincomb(1, rhs, -1, r)       # r = rhs - A x
    p = r.copy()
    Ap = op.domain.element() #Extra storage for storing A x
    
    sqnorm_r_old = r.norm()**2  # Only recalculate norm after update

    for _ in range(niter):
        op(p, out=Ap)  # Ap = A p
        
        alpha = sqnorm_r_old / p.inner(Ap)
        
        if alpha == 0.0:  # Return if residual is 0
            return
            
        x.lincomb(1, x, alpha, p)            # x = x + alpha*p
        r.lincomb(1, r, -alpha, Ap)           # r = r - alpha*p
        
        sqnorm_r_new = r.norm()**2    
        
        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p.lincomb(1, r, beta, p)                       # p = s + b * p

        if partial is not None:
            partial.send(x)

def conjugate_gradient_normal(op, x, rhs, niter=1, partial=None):
    """ Optimized version of CGN, uses no temporaries etc.
    """
    d = op(x)
    d.lincomb(1, rhs, -1, d)       # d = rhs - A x
    p = op.derivative(x).adjoint(d)
    s = p.copy()
    q = op.range.element()
    sqnorm_s_old = s.norm()**2  # Only recalculate norm after update

    for _ in range(niter):
        op(p, out=q)  # q = A p
        sqnorm_q = q.norm()**2
        if sqnorm_q == 0.0:  # Return if residual is 0
            return

        a = sqnorm_s_old / sqnorm_q
        x.lincomb(1, x, a, p)                       # x = x + a*p
        d.lincomb(1, d, -a, q)                      # d = d - a*Ap
        op.derivative(p).adjoint(d, out=s)  # s = A^T d

        sqnorm_s_new = s.norm()**2
        b = sqnorm_s_new / sqnorm_s_old
        sqnorm_s_old = sqnorm_s_new

        p.lincomb(1, s, b, p)                       # p = s + b * p

        if partial is not None:
            partial.send(x)


def exp_zero_seq(scale):
    """ The default zero sequence given by:
        t_m = scale ^ (-m-1)
    """
    value = 1.0
    while True:
        value /= scale
        yield value


def gauss_newton(op, x, rhs, niter=1, zero_seq=exp_zero_seq(2.0),
                 partial=None):
    """ Solves op(x) = rhs using the gauss newton method. The inner-solver
    uses conjugate gradient.
    """
    x0 = x.copy()
    I = IdentityOperator(op.domain)
    dx = x.space.zero()

    tmp_dom = op.domain.element()
    u = op.domain.element()
    tmp_ran = op.range.element()
    v = op.range.element()

    for _ in range(niter):
        tm = next(zero_seq)
        deriv = op.derivative(x)
        deriv_adjoint = deriv.adjoint

        # v = rhs - op(x) - deriv(x0-x)
        # u = deriv.T(v)
        op(x, out=tmp_ran)      # eval        op(x)
        v.lincomb(1, rhs, -1, tmp_ran)  # assign      v = rhs - op(x)
        tmp_dom.lincomb(1, x0, -1, x)  # assign temp  tmp_dom = x0 - x
        deriv(tmp_dom, out=tmp_ran)   # eval        deriv(x0-x)
        v -= tmp_ran                    # assign      v = rhs-op(x)-deriv(x0-x)
        deriv_adjoint(v, out=u)       # eval/assign u = deriv.T(v)

        # Solve equation system
        # (deriv.T o deriv + tm * I)^-1 u = dx
        A = OperatorSum(OperatorComp(deriv.adjoint, deriv),
                        tm * I, tmp_dom)

        # TODO: allow user to select other method
        conjugate_gradient(A, dx, u, 3)

        # Update x
        x.lincomb(1, x0, 1, dx)  # x = x0 + dx

        if partial is not None:
            partial.send(x)

class BacktrackingLineSearch(object):
    """ Backtracking line search, 
    a search scheme based on the Armijo-Goldstein condition.
    """
    def __init__(self, function, tau=0.8, c=0.7):
        self.function = function
        self.tau = tau
        self.c = c

    def __call__(self, x, direction, gradf):
        alpha = 1.0
        decrease = gradf.inner(direction)
        fx = self.function(x)
        while self.function(x + alpha * direction) >= fx + alpha * decrease * self.c:
            alpha *= self.tau
        return alpha

class ConstantLineSearch(object):
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, x, direction, gradf):
        return self.constant

def quasi_newton(op, x, line_search, niter=1, partial=None):
    """ General implementation of the quasi newton method for solving

    op(x) == 0
    """
    I = IdentityOperator(op.range)
    Bi = IdentityOperator(op.range)
    # Reusable temporaries
    for _ in range(niter):
        opx = op(x)
        print(opx.norm())
        p = Bi(-opx)
        alpha = line_search(x, p, opx)
        x_old = x.copy()
        s = alpha * p
        x += s
        y = op(x) - op(x_old)
        x_old = x
        ys = y.inner(s)

        if ys == 0.0:
            return

        Bi = (I - s * y.T / ys) * Bi *  (I - y * s.T / ys) + s * s.T / ys

        if partial is not None:
            partial.send(x)


def partial_derivative(f, axis=0, voxel_size=1.0, edge_order=2,
                       zero_padding=False):
    """Calculates the partial derivative of 'f' along direction of 'axis'.
    The number of voxels is maintained. Assuming (implicit) zero padding
    central differences are used on the interior and on endpoints. Otherwise
    either one-sided differences differences are used. In the latter case
    first-order accuracy can be triggered on endpoints with parameter
    'edge_order'. Assuming zero padding avoids inconsistencies with the
    'Gradient' operator and its adjoint 'Divergence' which occur when using
    one-sides differences on endpoint (no padding).

    >>> x = np.arange(10, dtype=float)
    >>> partial_derivative(x)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> partial_derivative(x, voxel_size=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    >>> partial_derivative(x, voxel_size=1.0, zero_padding=True)
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. , -4. ])
    >>> dx1 = partial_derivative(np.sin(x/10*np.pi), edge_order=1)
    >>> dx2 = partial_derivative(np.sin(x/10*np.pi), edge_order=2)
    >>> np.array_equal(dx1[1:-1], dx2[1:-1])
    True
    >>> dx1[0] == dx2[0]
    False
    >>> dx1[-1] == dx2[-1]
    False
    >>> n = 5
    >>> x = np.arange(n, dtype=float)
    >>> x = x * x.reshape((n,1))
    >>> partial_derivative(x, 0)
    array([[-0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.]])
    >>> partial_derivative(x, 1)
    array([[-0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.,  3.],
           [ 4.,  4.,  4.,  4.,  4.]])
    >>> try:
    ...    partial_derivative(x, 2)
    ... except IndexError, e:
    ...    print(e)
    Axis paramater (2) exceeds number of dimensions (2).
    """

    f_data = np.asanyarray(f)
    nd = f_data.ndim

    # create slice objects --- initially all are [:, :, ..., :]
    # noinspection PyTypeChecker
    slice1 = [slice(None)] * nd
    # noinspection PyTypeChecker
    slice2 = [slice(None)] * nd
    # noinspection PyTypeChecker
    slice3 = [slice(None)] * nd
    # noinspection PyTypeChecker
    slice4 = [slice(None)] * nd

    try:
        if f_data.shape[axis] < 2:
            raise ValueError("Shape of array too small to calculate a "
                             "numerical gradient, at least two elements are "
                             "required.")
    except IndexError:
        raise IndexError("Axis paramater ({0}) exceeds number of dimensions "
                         "({1}).".format(axis, nd))

    out = np.empty_like(f_data)

    # Numerical differentiation: 2nd order interior
    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(2, None)
    # noinspection PyTypeChecker
    slice3[axis] = slice(None, -2)
    # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
    out[slice1] = (f_data[slice2] - f_data[slice3]) / 2.0

    # central differences
    if zero_padding:
        # Assume zeros for indices outside the volume

        # 1D equivalent -- out[0] = (y[1] - 0)/2.0
        slice1[axis] = 0
        slice2[axis] = 1
        out[slice1] = f_data[slice2] / 2.0

        # 1D equivalent -- out[-1] = (0 - y[-2])/2.0
        slice1[axis] = -1
        slice3[axis] = -2
        out[slice1] = - f_data[slice3] / 2.0

    # one-side differences
    else:
        # Numerical differentiation: 1st order edges
        if f_data.shape[axis] == 2 or edge_order == 1:

            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            # 1D equivalent -- out[0] = (y[1] - y[0])
            out[slice1] = (f_data[slice2] - f_data[slice3])

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            out[slice1] = (f_data[slice2] - f_data[slice3])

        # Numerical differentiation: 2nd order edges
        else:

            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            # 1D equivalent -- out[0] = -(3*y[0] - 4*y[1] + y[2]) / 2.0
            out[slice1] = -(3.0 * f_data[slice2] - 4.0 * f_data[slice3] +
                            f_data[slice4]) / 2.0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            slice4[axis] = -3
            # 1D equivalent -- out[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / 2.0
            out[slice1] = (3.0 * f_data[slice2] - 4.0 * f_data[slice3] +
                           f_data[slice4]) / 2.0

    # divide by step size
    out /= voxel_size

    return out

# noinspection PyAbstractClass
class Gradient(Operator):
    """Gradient operator for any number of dimension. Calls function
    'partial_derivative' to calculate each component.
    """

    def __init__(self, space, voxel_size=(1,), edge_order=2,
                 zero_padding=True):
        self.voxel_size = voxel_size
        self.edge_order = edge_order
        self.zero_padding = zero_padding
        super().__init__(domain=space, range=ProductSpace(
            space, len(voxel_size)), linear=True)

    def _apply(self, rhs, out):
        """Apply gradient operator to 'rhs' and store result in 'out'.

        >>> from odl import *
        >>> from utils import ndvolume
        >>> N = 3
        >>> n = 10
        >>> disc = l2_uniform_discretization(L2(IntervalProd(
        ... [0.]*N, [n]*N)), [n]*N)
        >>> x = disc.element(ndvolume(n, N, np.int16))
        >>> print(x.space)
        >>> A = Gradient(disc, (1.,)*N, zero_padding=True)
        >>> Ax = A(x)
        >>> g = A.range.element((1.,)*N)
        >>> Adg = A.adjoint(g)
        >>> g.inner(Ax) - x.inner(Adg)
        0.0
        >>> B = Divergence(disc, (1.,)*N, zero_padding=True)
        >>> Bg = B(g)
        >>> Bdx = B.adjoint(x)
        """
        rhs_data = np.asanyarray(rhs)
        nd = rhs_data.ndim

        dx = self.voxel_size
        if np.size(dx) == 1:
            dx = [dx for _ in range(nd)]

        for axis in range(nd):
            out[axis][:] = partial_derivative(rhs_data, axis, dx[axis],
                                              self.edge_order,
                                              self.zero_padding)

    @property
    def adjoint(self):
        """Note that, the first argument of the 'Divergence' operator is the
        space the gradient is computed and not its domain. Thus, 'Divergence'
        takes the domain of 'Gradient' as space argument.
        """
        return -Divergence(self.domain, voxel_size=self.voxel_size,
                           edge_order=self.edge_order,
                           zero_padding=self.zero_padding)


# noinspection PyAbstractClass
class Divergence(Operator):
    """Divergence operator for any number of dimensions. Calls function
    'partial_derivative' for each component of the input vector. Using
    'zero_padding' '-Divergence' is the adjoint of 'Gradient'.
    """
    def __init__(self, space, voxel_size=(1,), edge_order=2,
                 zero_padding=True):
        self.space = space
        self.voxel_size = voxel_size
        self.edge_order = edge_order
        self.zero_padding = zero_padding
        super().__init__(domain=ProductSpace(space, len(voxel_size)),
                         range=space, linear=True)

    def _apply(self, rhs, out):
        """Apply 'Divergence' operator to 'rhs' and store result in 'out'."""

        tmp = np.zeros_like(rhs[0].asarray())
        for axis in range(tmp.ndim):
            # tmp += self._partial(rhs[nn].asarray(), nn)
            tmp += partial_derivative(rhs[axis].asarray(), axis=axis,
                                      voxel_size=self.voxel_size[axis],
                                      edge_order=self.edge_order,
                                      zero_padding=self.zero_padding)
        out[:] = tmp

    @property
    def adjoint(self):
        return -Gradient(self.range, voxel_size=self.voxel_size,
                         edge_order=self.edge_order,
                         zero_padding=self.zero_padding)


def operator_norm(op, niter=1, x_init=1.0, precision=None, partial=None):
    """Calulates the norm '||K||_2' of operator 'K' as the largest singular
    value of 'K' employing the generic power method.  The obtained scalar
    tends to '||K||_2' as the number of iterations 'niter' increases. Loop
    is aborted after 'niter' iterations or if consecutive estimates of the
    norm differ less than 'precision'.

    :param op: continuous linear operator
    :type niter: int (default 1)
    :param niter: number of iteration for the generic power method
    :param x_init: initial non-zero image in the domain of the operator
    :type precision: float
    :param precision: abort loop if operator changes less than precision
    :rtype: float
    :returns: s

    >>> from odl import *
    >>> N = 2
    >>> n = 10
    >>> disc = l2_uniform_discretization(L2(Cuboid([0.]*N, [n]*N)), [n]*N)
    >>> x = disc.element(1)
    >>> op = Gradient(disc, (1,)*N, zero_padding=True)
    >>> op_norms = StorePartial()
    >>> op_norm = operator_norm(op, 20, 1.0, partial=op_norms)
    >>> op_norms.results[-1] - op_norm
    0.0
    >>> print('{0:.5f} {1:.5f}'.format(*op_norms.results[-2:]))
    1.62585 1.63068
    >>> print('{:.5f}'.format(operator_norm(op, 100, 1.0, precision=1e-5)))
    1.66184
    >>> op_norms = StorePartial()
    >>> prec = 1e-5
    >>> op_norm = operator_norm(op, 100, 1.0, precision=prec, partial=op_norms)
    >>> op_norms.results[-1] - op_norms.results[-2] < prec
    True
    >>> op_norms.results[-2] - op_norms.results[-3] > prec
    True
    >>> op_norm == operator_norm(op, len(op_norms.results), 1.0)
    True
    """

    x = op.domain.element(x_init)
    tmp_ran = op.range.element()
    s0 = 0

    for _ in range(niter):
        # x_{n+1} <- K^T K x_n
        op(x, out=tmp_ran)
        op.adjoint(tmp_ran, out=x)
        # x_n <- x_n/||x_n||_2
        x = x / x.norm()

        # intermediate results
        if partial or precision is not None:
            # s <-|| K x ||_2
            op(x, out=tmp_ran)
            s = tmp_ran.norm()

        if partial is not None:
            partial.send(s)

        if precision is not None and niter > 2:
            if abs(s - s0) < precision:
                break
            s0 = s

    # s <-|| K x ||_2
    op(x, out=tmp_ran)
    return tmp_ran.norm()


def chambolle_pock(op, x, rhs, niter=1, op_norm=None, tau=None, sigma=None,
                   theta=None, partial=None):
    """Chambolle-Pock algorithms of first-order primal-dual method for
    non-smooth convex optimizatoin problems with known saddle point structure.


    Parameters
    ----------
    :param op: instance of 'Operator' subclass
    :param op: continuous linear operator with induced norm
    :param x: odl vector
    :param rhs: odl vector
    :type niter: int (default 1)
    :param niter: number of iterations
    :type op_norm: float (default: None)
    :param op_norm: operator norm. If 'None' generic power method is used to
        calculate the operator norm.
    :type tau: float (default 1/op_norm)
    :param tau: step size
    :type sigma: float (default 1/op_norm)
    :param sigma: step size
    :type theta: float (default 1)
    :param theta: acceleration parameter in [0,1]. theta = 0 corresponds to
    the Arrow-Hurwicz algorithm.
    :param partial:

    :return:
    """

    # step 0
    u = x
    ub = u.copy()
    g = rhs

    # step 1:
    if op_norm is None:
        op_norm = operator_norm(op, 20)
    if tau is None:
        tau = 0.95 / op_norm
    if sigma is None:
        sigma = 0.95 / op_norm
    if theta is None:
        theta = 1

    # Reusable temporaries
    p = op.range.element(0)
    p_tmp = op.range.element()

    for _ in range(niter):
        # step 5: p_{n+1} <- (p_n + sigma(A^T ub_n - g)) / (1 + sigma)
        op(ub, out=p_tmp)
        p_tmp -= g
        p_tmp *= sigma
        p += p_tmp
        p /= 1+sigma

        # step 6: u_{n+1} <- u_{n} - tau * A^T p_{n+1}
        # Store current u_n in ub
        ub = u.copy()
        op.adjoint(p, out=u)
        u *= -tau
        # u_{n+1}
        u += ub

        if partial is not None:
            partial.send(u)

        # step 7: ub_{n+1} <- u_{n+1} + theta(u_{n+1} - u_n)
        ub *= -1
        ub += u
        ub *= theta
        ub += u

    x = u

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
