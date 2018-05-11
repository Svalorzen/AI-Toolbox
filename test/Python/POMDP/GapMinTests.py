import unittest
import sys
import os

sys.path.append(os.getcwd())

from AIToolbox import POMDP

class POMDPPythonGapMin(unittest.TestCase):

    def chengD35(self):
        # Actions are: 0-listen, 1-open-left, 2-open-right
        S = 3
        A = 3
        O = 3

        model = POMDP.Model(O, S, A)

        # SAS form
        t = [0,0,0]
        r = [0,0,0]
        # SAO form
        o = [0,0,0]

        t[0] = [
            [0.445, 0.222, 0.333],
            [0.234, 0.064, 0.702],
            [0.535, 0.313, 0.152],
        ]

        t[1] = [
            [0.500, 0.173, 0.327],
            [0.549, 0.218, 0.233],
            [0.114, 0.870, 0.016],
        ]

        t[2] = [
            [0.204, 0.553, 0.243],
            [0.061, 0.466, 0.473],
            [0.325, 0.360, 0.315],
        ]

        o[0] = [
            [0.686, 0.182, 0.132],
            [0.698, 0.131, 0.171],
            [0.567, 0.234, 0.199],
        ]

        o[1] = [
            [0.138, 0.786, 0.076],
            [0.283, 0.624, 0.093],
            [0.243, 0.641, 0.116],
        ]

        o[2] = [
            [0.279, 0.083, 0.638],
            [0.005, 0.202, 0.793],
            [0.186, 0.044, 0.770],
        ]

        r[0] = [
            [5.2] * 3,
            [0.8] * 3,
            [9.0] * 3,
        ]

        r[1] = [
            [4.6] * 3,
            [6.8] * 3,
            [9.3] * 3,
        ]

        r[2] = [
            [4.1] * 3,
            [6.9] * 3,
            [0.8] * 3,
        ]

        model.setTransitionFunction(t)
        model.setRewardFunction(r)
        model.setObservationFunction(o)
        model.setDiscount(0.999)

        return model

    def test_solver(self):
        gm = POMDP.GapMin(0.005, 3);

        model = self.chengD35();

        initialBelief = [1.0/3, 1.0/3, 1.0/3]

        lb, ub, vlist, qfun = gm(model, initialBelief);

        self.assertTrue(9.0 < ub - lb and ub - lb < 11.0);

if __name__ == '__main__':
    unittest.main(verbosity=2)
