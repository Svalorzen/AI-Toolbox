#define BOOST_TEST_MODULE MDP_PrioritizedSweeping
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/PrioritizedSweeping.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/MDP/Environments/CliffProblem.hpp>

BOOST_AUTO_TEST_CASE( cliff ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(12, 3);

    Model model = makeCliffProblem(grid);

    Experience exp(model.getS(), model.getA());
    RLModel<Experience> learnedModel(exp, 1.0, false);

    PrioritizedSweeping solver(learnedModel);

    QGreedyPolicy gPolicy(solver.getQFunction());
    EpsilonPolicy ePolicy(gPolicy, 0.1);

    size_t start = model.getS() - 2;

    size_t s, a;

    for ( int episode = 0; episode < 10; ++episode ) {
        s = start;
        for ( int i = 0; i < 10000; ++i ) {
            a = ePolicy.sampleAction( s );
            const auto [s1, rew] = model.sampleSR( s, a );

            exp.record(s, a, s1, rew);
            learnedModel.sync(s, a, s1);

            solver.stepUpdateQ(s, a);
            solver.batchUpdateQ();

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
