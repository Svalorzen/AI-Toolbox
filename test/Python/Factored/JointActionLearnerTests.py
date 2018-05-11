import unittest
import sys
import os

sys.path.append(os.getcwd())
from AIToolbox import Factored

class FactoredPythonJointActionLearnerTests(unittest.TestCase):

    def testSimple(self):
        S = 3
        A = [2,2,2]
        agentId = 0

        discount = 0.9
        learningRate = 0.1

        l = Factored.MDP.JointActionLearner(S, A, agentId, discount, learningRate)

        a = [0,0,0]

        l.stepUpdateQ(0, a, 1, 10.0)

        self.assertEqual(l.getSingleQFunction()[0,0], 1.0)

        l.stepUpdateQ(0, a, 1, 10.0)

        a[1] = 1;
        l.stepUpdateQ(0, a, 1, 6.0)

        self.assertEqual(l.getSingleQFunction()[0,0], (1.9 * 2.0 + 0.6) / 3.0)

        l.stepUpdateQ(2, a, 0, 10.0)

        self.assertEqual(l.getSingleQFunction()[2,0], 1.0 + 0.09 * 1.9)

if __name__ == '__main__':
    unittest.main(verbosity=2)

