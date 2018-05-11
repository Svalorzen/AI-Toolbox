import unittest
import sys
import os

sys.path.append(os.getcwd())
from AIToolbox import MDP

class MDPPythonQLearningTests(unittest.TestCase):

    def testUpdates(self):
        solver = MDP.QLearning(5, 5, 0.9, 0.5)

        # State goes to itself, thus needs to consider
        # next-step value.
        solver.stepUpdateQ(0, 0, 0, 10)
        self.assertEqual( solver.getQFunction()[0, 0], 5.0 )

        solver.stepUpdateQ(0, 0, 0, 10)
        self.assertEqual( solver.getQFunction()[0, 0], 9.75 )

        # Here it does not, so improvement is slower.
        solver.stepUpdateQ(3, 0, 4, 10)
        self.assertEqual( solver.getQFunction()[3, 0], 5.0 )

        solver.stepUpdateQ(3, 0, 4, 10)
        self.assertEqual( solver.getQFunction()[3, 0], 7.50 )

        # Test that index combinations are right.
        solver.stepUpdateQ(0, 1, 1, 10)
        self.assertEqual( solver.getQFunction()[0, 1], 5.0  )
        self.assertEqual( solver.getQFunction()[1, 0], 0.0  )
        self.assertEqual( solver.getQFunction()[1, 1], 0.0  )

if __name__ == '__main__':
    unittest.main(verbosity=2)
