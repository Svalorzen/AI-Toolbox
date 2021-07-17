import unittest
import sys
import os
from builtins import range

import pickle
import tempfile

sys.path.append(os.getcwd())

from AIToolbox import POMDP
from Utils.TigerProblem import *

class POMDPPythonSparseModelTests(unittest.TestCase):

    def testDefaultBuild(self):
        m = POMDP.SparseModel(3,5,4,0.95)

        self.assertEqual(m.getS(), 5)
        self.assertEqual(m.getA(), 4)
        self.assertEqual(m.getO(), 3)
        self.assertEqual(m.getDiscount(), 0.95)

    def testCopy(self):
        model = makeTigerProblem()
        sparseModel = POMDP.SparseModel(model)

        self.assertEqual(model.getS(), sparseModel.getS())
        self.assertEqual(model.getA(), sparseModel.getA())
        self.assertEqual(model.getO(), sparseModel.getO())
        self.assertEqual(model.getDiscount(), sparseModel.getDiscount())

        for s in range(model.getS()):
            for a in range(model.getA()):
                for s1 in range(model.getS()):
                    self.assertAlmostEqual(
                        model.getTransitionProbability(s,a,s1),
                        sparseModel.getTransitionProbability(s,a,s1)
                    )
                    self.assertAlmostEqual(
                        model.getExpectedReward(s,a,s1),
                        sparseModel.getExpectedReward(s,a,s1)
                    )

        for s in range(model.getS()):
            for a in range(model.getA()):
                for o in range(model.getO()):
                    self.assertAlmostEqual(
                        model.getObservationProbability(s,a,o),
                        sparseModel.getObservationProbability(s,a,o)
                    )

    def testPickle(self):
        model = POMDP.SparseModel(makeTigerProblem())
        model.setDiscount(0.95)

        with tempfile.TemporaryFile() as fp:
            pickle.dump(model, fp)
            fp.seek(0)
            newModel = pickle.load(fp)

        self.assertEqual(model.getS(), newModel.getS())
        self.assertEqual(model.getA(), newModel.getA())
        self.assertEqual(model.getO(), newModel.getO())
        self.assertEqual(model.getDiscount(), newModel.getDiscount())

        for s in range(model.getS()):
            for a in range(model.getA()):
                for o in range(model.getO()):
                    self.assertAlmostEqual(
                        model.getObservationProbability(s,a,o),
                        newModel.getObservationProbability(s,a,o)
                    )

if __name__ == '__main__':
    unittest.main(verbosity=2)
