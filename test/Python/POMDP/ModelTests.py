import unittest
import sys
import os
from builtins import range

import pickle
import tempfile

sys.path.append(os.getcwd())

from AIToolbox import POMDP
from Utils.TigerProblem import *

class POMDPPythonModelTests(unittest.TestCase):

    def testDefaultBuild(self):
        m = POMDP.Model(3,5,4,0.95)

        self.assertEqual(m.getS(), 5)
        self.assertEqual(m.getA(), 4)
        self.assertEqual(m.getO(), 3)
        self.assertEqual(m.getDiscount(), 0.95)

    def testPickle(self):
        model = makeTigerProblem()
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
