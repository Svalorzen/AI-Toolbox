#define BOOST_TEST_MODULE MDP_WoLFPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Policies/WoLFPolicy.hpp>
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

    MDP::QLearning solver(S, A, 1.0, 1.0);
    MDP::QLearning solver2(S, A, 1.0, 1.0);
    // This is important: the two policies must be different somehow (maybe
    // even starting from a different initial policy would suffice, although I
    // have no idea), otherwise it seems that learning does not converge at
    // all; this is because the two policies seem to change completely in
    // unison, so they never get to converge. Weird..
    MDP::WoLFPolicy policy(solver.getQFunction());
    MDP::WoLFPolicy policy2(solver2.getQFunction(), 0.1, 0.5);

    // Other important thing: without exploration, the policies cannot
    // explore enough and won't end up converging. So we need to wrap
    // the policies in EpsilonPolicy in order to get good exploration.
    MDP::EpsilonPolicy p(policy);
    MDP::EpsilonPolicy p2(policy2);

    for ( unsigned i = 0; i < 150000; ++i ) {
        size_t a = p.sampleAction(0);
        size_t b = p2.sampleAction(0);

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

    MDP::WoLFPolicy policy(solver.getQFunction());
    MDP::WoLFPolicy policy2(solver2.getQFunction());
    // Here they're not really needed since there's only
    // two actions, but just to do things properly.
    MDP::EpsilonPolicy p(policy);
    MDP::EpsilonPolicy p2(policy2);

    for ( unsigned i = 0; i < 150000; ++i ) {
        size_t a = p.sampleAction(0);
        // Self-play, b is opponent
        size_t b = p2.sampleAction(0);

        int result = ( a == b ) ? 1 : -1;

        solver.stepUpdateQ(0, a, 0, result);
        solver2.stepUpdateQ(0, b, 0, -result);

        policy.stepUpdateP(0);
        policy2.stepUpdateP(0);
    }

    BOOST_CHECK(policy.getActionProbability(0,0) < 0.6);
    BOOST_CHECK(policy.getActionProbability(0,0) > 0.4);
}
