#define BOOST_TEST_MODULE MDP_QL
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Algorithms/QL.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/Bandit/Policies/RandomPolicy.hpp>
#include <AIToolbox/MDP/Policies/BanditPolicyAdaptor.hpp>

#include <AIToolbox/MDP/Environments/CliffProblem.hpp>

BOOST_AUTO_TEST_CASE( cliff ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(12, 3);

    auto model = makeCliffProblem(grid);

    BanditPolicyAdaptor<AIToolbox::Bandit::RandomPolicy> behaviour(model.getS(), model.getA());
    QL solver(behaviour.getS(), behaviour.getA());

    QGreedyPolicy gPolicy(solver.getQFunction());

    size_t start = model.getS() - 2;

    constexpr auto episodes = 3000;
    size_t s, a;
    for ( int episode = 0; episode < episodes; ++episode ) {
        solver.setEpsilon(0.1 - (0.1 / episodes) * episode);
        s = start;
        for ( int i = 0; i < 10000; ++i ) {
            a = behaviour.sampleAction( s );
            const auto [s1, rew] = model.sampleSR( s, a );

            solver.stepUpdateQ( s, a, s1, rew );

            if ( s1 == model.getS() - 1 ) break;
            s = s1;
        }
    }

    // Final path should be: UPx1, RIGHTx11, DOWNx1. Total moves: 13
    // We can use states only from above the cliff though
    BOOST_CHECK_EQUAL( gPolicy.getActionProbability(start, UP), 1.0 );

    auto state = grid(0, 2);
    for ( int i = 0; i < 11; ++i ) {
        BOOST_CHECK_EQUAL( gPolicy.getActionProbability(state, RIGHT), 1.0 );
        state = grid.getAdjacent(RIGHT, state);
    }
    for ( int i = 0; i < 1; ++i ) {
        BOOST_CHECK_EQUAL( gPolicy.getActionProbability(state, DOWN), 1.0 );
        state = grid.getAdjacent(DOWN, state);
    }
}
