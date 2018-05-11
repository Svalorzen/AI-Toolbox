import unittest
import sys
import os

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
        self.assertEqual(exp.getReward(0,0,0), 0.0)

        self.assertEqual(exp.getVisits(S-1,A-1,S-1), 0)
        self.assertEqual(exp.getReward(S-1,A-1,S-1), 0.0)

    def testRecording(self):
        S , A = 5, 6
        exp = MDP.Experience(S, A)

        s, s1, a = 3, 4, 5
        rew, negrew, zerorew = 7.4, -4.2, 0.0

        self.assertEqual(exp.getVisits(s,a,s1), 0)

        exp.record(s,a,s1,rew)

        self.assertEqual(exp.getVisits(s,a,s1), 1)
        self.assertEqual(exp.getReward(s,a,s1), rew)

        exp.reset()

        self.assertEqual(exp.getVisits(s,a,s1), 0)

        exp.record(s,a,s1,negrew)

        self.assertEqual(exp.getVisits(s,a,s1), 1)
        self.assertEqual(exp.getReward(s,a,s1), negrew)

        exp.record(s,a,s1,zerorew)

        self.assertEqual(exp.getVisits(s,a,s1), 2)
        self.assertEqual(exp.getReward(s,a,s1), negrew)

        self.assertEqual(exp.getVisitsSum(s, a), 2)

    def testCompatibility(self):
        S, A = 4, 3
        exp = MDP.Experience(S, A)

        visits = []
        rewards = []
        for s in xrange(0, S):
            visits.append([])
            rewards.append([])
            for a in xrange(0, A):
                visits[s].append([])
                rewards[s].append([])
                for s1 in xrange(0, S):
                    visits[s][a].append(generator())
                    rewards[s][a].append(generator())

        exp.setVisits(visits);
        exp.setRewards(rewards);

        for s in xrange(0, S):
            for a in xrange(0, A):
                visitsSum, rewardSum = 0, 0
                for s1 in xrange(0, S):
                    self.assertEqual( exp.getVisits(s,a,s1), visits[s][a][s1] );
                    self.assertEqual( exp.getReward(s,a,s1), rewards[s][a][s1] );
                    visitsSum += visits[s][a][s1];
                    rewardSum += rewards[s][a][s1];

                self.assertEqual( exp.getVisitsSum(s,a), visitsSum );
                self.assertEqual( exp.getRewardSum(s,a), rewardSum );

if __name__ == '__main__':
    unittest.main(verbosity=2)
