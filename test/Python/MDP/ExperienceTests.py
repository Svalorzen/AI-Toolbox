import unittest
import sys
import os
from builtins import range

sys.path.append(os.getcwd())
from AIToolbox import MDP

def generator():
    generator.counter += 1
    return generator.counter;

generator.counter = 0

class MDPPythonExperienceTests(unittest.TestCase):

    def testConstruction(self):
        S, A = 5, 6
        exp = MDP.Experience(S, A)

        self.assertEqual(exp.getS(), S)
        self.assertEqual(exp.getA(), A)
        self.assertEqual(exp.getVisits(0,0,0), 0)
        self.assertEqual(exp.getReward(0,0), 0.0)

        self.assertEqual(exp.getVisits(S-1,A-1,S-1), 0)
        self.assertEqual(exp.getReward(S-1,A-1), 0.0)

    def testRecording(self):
        S , A = 5, 6
        exp = MDP.Experience(S, A)

        s, s1, a = 3, 4, 5
        rew, negrew, zerorew = 7.4, -4.2, 0.0

        self.assertEqual(exp.getVisits(s,a,s1), 0)

        exp.record(s,a,s1,rew)

        self.assertEqual(exp.getVisits(s,a,s1), 1)
        self.assertEqual(exp.getReward(s,a), rew)

        exp.reset()

        self.assertEqual(exp.getVisits(s,a,s1), 0)
        self.assertEqual(exp.getReward(s,a), 0)

        exp.record(s,a,s1,negrew)

        self.assertEqual(exp.getVisits(s,a,s1), 1)
        self.assertEqual(exp.getReward(s,a), negrew)

        exp.record(s,a,s1,zerorew)

        self.assertEqual(exp.getVisits(s,a,s1), 2)
        self.assertEqual(exp.getReward(s,a), negrew / 2.0)

        self.assertEqual(exp.getVisitsSum(s, a), 2)

    def testCompatibility(self):
        S, A = 4, 3
        exp = MDP.Experience(S, A)

        visits = []
        rewards = []
        for s in range(0, S):
            visits.append([])
            rewards.append([])
            for a in range(0, A):
                rewards[s].append(generator())
                visits[s].append([])
                for s1 in range(0, S):
                    visits[s][a].append(generator())

        exp.setVisitsTable(visits);
        exp.setRewardMatrix(rewards);

        for s in range(0, S):
            for a in range(0, A):
                visitsSum = 0
                for s1 in range(0, S):
                    self.assertEqual( exp.getVisits(s,a,s1), visits[s][a][s1] );
                    visitsSum += visits[s][a][s1];

                self.assertEqual( exp.getVisitsSum(s,a), visitsSum );
                self.assertEqual( exp.getReward(s,a), rewards[s][a] );

if __name__ == '__main__':
    unittest.main(verbosity=2)
