# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import
from future import standard_library
try:
    from builtins import range
except ImportError:
    from future.builtins import range

# External module imports
import unittest
from math import pi
import numpy as np

# RL imports
from RL.operator.operator import *
from RL.space.euclidean import EuclidianSpace
import RL.space.discretizations as disc
import RL.space.function as fs
import RL.space.set as sets
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class L2Test(RLTestCase):
    def testInterval(self):
        I = sets.Interval(0, pi)
        l2 = fs.L2(I)
        rn = EuclidianSpace(10)
        d = disc.makeUniformDiscretization(l2, rn)

        l2sin = l2.makeVector(np.sin)
        sind = d.makeVector(l2sin)

        self.assertAlmostEqual(sind.normSq(), pi/2)

    def testRectangle(self):
        R = sets.Rectangle((0, 0), (pi, 2*pi))
        l2 = fs.L2(R)
        n = 10
        m = 10
        rn = EuclidianSpace(n*m)
        d = disc.makePixelDiscretization(l2, rn, n, m)

        l2sin = l2.makeVector(lambda point: np.sin(point[0]) *
                              np.sin(point[1]))
        sind = d.makeVector(l2sin)

        self.assertAlmostEqual(sind.normSq(), pi**2 / 2)

#    def testCubes(self):
#        dmax = 6
#        for dim in range(3, dmax):
#            start = [0] * dim
#            end = [2*pi] * dim
#            cube = sets.IntervalProd(start, end)
#            l2 = fs.L2(cube)
#            rn = EuclidianSpace(10**dim)
#            discr = disc.makePixelDiscretization

if __name__ == '__main__':
    unittest.main(exit=False)
