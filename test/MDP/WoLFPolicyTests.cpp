#define BOOST_TEST_MODULE MDP_WoLFPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Policies/WoLFPolicy.hpp>

#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>
#include <AIToolbox/MDP/Algorithms/PrioritizedSweeping.hpp>

#include <random>
#include <fstream>

int testRockPaperScissors(unsigned a, unsigned b) {
    if ( a == b ) return 0;
    if ( a == ( (b+1)%3 ) ) return 1;
    return -1;
}

BOOST_AUTO_TEST_CASE( rock_paper_scissors_random ) {
    using namespace AIToolbox;
    size_t S = 1, A = 3;

    Experience exp(S,A);
    MDP::RLModel model(exp, 1.0, false);
    MDP::PrioritizedSweeping<MDP::RLModel> solver(model);

    MDP::WoLFPolicy policy(solver.getQFunction());

    std::default_random_engine engine(12345);
    std::uniform_int_distribution<unsigned> dist(0,2);

    for ( unsigned i = 0; i < 100000; ++i ) {
        size_t a = policy.sampleAction(0);
        size_t b = policy.sampleAction(0);

        int result = testRockPaperScissors(a,b);

        exp.record(0,a,0,result);
        model.sync(0,a);
        solver.stepUpdateQ(0,a);
        solver.batchUpdateQ();

        policy.updatePolicy(0);
    }

    BOOST_CHECK(policy.getActionProbability(0,0) < 0.43);
    BOOST_CHECK(policy.getActionProbability(0,0) > 0.23);

    BOOST_CHECK(policy.getActionProbability(0,1) < 0.43);
    BOOST_CHECK(policy.getActionProbability(0,1) > 0.23);
}

BOOST_AUTO_TEST_CASE( matching_pennies ) {
    using namespace AIToolbox;
    size_t S = 1, A = 2;

    Experience exp(S,A);
    MDP::RLModel model(exp, 1.0, false);
    MDP::PrioritizedSweeping<MDP::RLModel> solver(model);

    MDP::WoLFPolicy policy(solver.getQFunction());

    std::default_random_engine engine(12345);
    std::uniform_int_distribution<unsigned> dist(0,1);

    for ( unsigned i = 0; i < 100000; ++i ) {
        size_t a = policy.sampleAction(0);

        int result = ( a == dist(engine) ) ? 1 : -1;

        exp.record(0,a,0,result);
        model.sync(0,a);
        solver.stepUpdateQ(0,a);
        solver.batchUpdateQ();

        policy.updatePolicy(0);
    }

    BOOST_CHECK(policy.getActionProbability(0,0) < 0.6);
    BOOST_CHECK(policy.getActionProbability(0,0) > 0.4);
}
