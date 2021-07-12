import unittest
import sys
import os
from builtins import range
import pickle
import tempfile

sys.path.append(os.getcwd())

from AIToolbox import POMDP
from Utils.TigerProblem import *

class POMDPPythonPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = makeTigerProblem()
        self.model.setDiscount(0.95)

        self.horizon = 15
        solver = POMDP.IncrementalPruning(self.horizon, 0.0)
        solution = solver(self.model)

        self.vf = solution[1]

    def testDefaultBuild(self):
        policy = POMDP.Policy(3, 4, 5)

        self.assertEqual(policy.getS(), 3)
        self.assertEqual(policy.getA(), 4)
        self.assertEqual(policy.getO(), 5)

    def testVFBuild(self):
        policy = POMDP.Policy(
                self.model.getS(),
                self.model.getA(),
                self.model.getO(),
                self.vf
        )

        belief = [0.0, 1.0]
        policy.sampleAction(belief, self.horizon)

        self.assertEqual(policy.getS(), self.model.getS())
        self.assertEqual(policy.getA(), self.model.getA())
        self.assertEqual(policy.getO(), self.model.getO())

    def testPickle(self):
        policy = POMDP.Policy(
                self.model.getS(),
                self.model.getA(),
                self.model.getO(),
                self.vf
        )

        with tempfile.TemporaryFile() as fp:
            pickle.dump(policy, fp)
            fp.seek(0)
            newPolicy = pickle.load(fp)

        self.assertEqual(policy.getS(), newPolicy.getS())
        self.assertEqual(policy.getA(), newPolicy.getA())
        self.assertEqual(policy.getO(), newPolicy.getO())

        belief = [0.0, 1.0]
        self.assertEqual(
            policy.sampleAction(belief, self.horizon),
            newPolicy.sampleAction(belief, self.horizon)
        )

        belief = [1.0, 0.0]
        self.assertEqual(
            policy.sampleAction(belief, self.horizon),
            newPolicy.sampleAction(belief, self.horizon)
        )

        belief = [0.5, 0.5]
        self.assertEqual(
            policy.sampleAction(belief, self.horizon),
            newPolicy.sampleAction(belief, self.horizon)
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
