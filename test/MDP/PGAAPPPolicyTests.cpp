#define BOOST_TEST_MODULE MDP_PGAAPPPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Policies/PGAAPPPolicy.hpp>
#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>

#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

int testRockPaperScissors(unsigned a, unsigned b) {
    if ( a == b ) return 0;
    if ( a == ( (b+1)%3 ) ) return 1;
    return -1;
}

BOOST_AUTO_TEST_CASE( rock_paper_scissors_random ) {
    using namespace AIToolbox;
    size_t S = 1, A = 3;

    MDP::QLearning solver(S, A, 0.0001, 0.8);
    MDP::QLearning solver2(S, A, 0.0001, 0.8);

    MDP::PGAAPPPolicy policy(solver.getQFunction());
    MDP::PGAAPPPolicy policy2(solver2.getQFunction());

    // Other important thing: without exploration, the policies cannot
    // explore enough and won't end up converging. So we need to wrap
    // the policies in EpsilonPolicy in order to get good exploration.
    MDP::EpsilonPolicy p(policy, 0.05);
    MDP::EpsilonPolicy p2(policy2, 0.05);

    for ( unsigned i = 0; i < 150000; ++i ) {
        size_t a = p.sampleAction(0);
        size_t b = p2.sampleAction(0);

        double learningRate = 5.0/(5000.0 + i);
        policy.setLearningRate(learningRate);
        policy2.setLearningRate(learningRate);

        int result = testRockPaperScissors(a,b);

        solver.stepUpdateQ(0, a, 0, result);
        solver2.stepUpdateQ(0, b, 0, -result);

        policy.stepUpdateP(0);
        policy2.stepUpdateP(0);
    }

    BOOST_CHECK(policy.getActionProbability(0,0) < 0.4333);
    BOOST_CHECK(policy.getActionProbability(0,0) > 0.2333);

    BOOST_CHECK(policy.getActionProbability(0,1) < 0.4333);
    BOOST_CHECK(policy.getActionProbability(0,1) > 0.2333);
}

BOOST_AUTO_TEST_CASE( matching_pennies ) {
    using namespace AIToolbox;
    size_t S = 1, A = 2;

    MDP::QLearning solver(S, A);
    MDP::QLearning solver2(S, A);

    MDP::PGAAPPPolicy policy(solver.getQFunction());
    MDP::PGAAPPPolicy policy2(solver2.getQFunction());
    // Here they're not really needed since there's only
    // two actions, but just to do things properly.
    MDP::EpsilonPolicy p(policy, 0.05);
    MDP::EpsilonPolicy p2(policy2, 0.05);

    for ( unsigned i = 0; i < 150000; ++i ) {
        size_t a = p.sampleAction(0);
        size_t b = p2.sampleAction(0);

        double learningRate = 5.0/(5000.0 + i);
        policy.setLearningRate(learningRate);
        policy2.setLearningRate(learningRate);

        int result = ( a == b ) ? 1 : -1;

        solver.stepUpdateQ(0, a, 0, result);
        solver2.stepUpdateQ(0, b, 0, -result);

        policy.stepUpdateP(0);
        policy2.stepUpdateP(0);
    }

    BOOST_CHECK(policy.getActionProbability(0,0) < 0.6);
    BOOST_CHECK(policy.getActionProbability(0,0) > 0.4);
}
