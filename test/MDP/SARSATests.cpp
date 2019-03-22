#define BOOST_TEST_MODULE MDP_SARSA
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Algorithms/SARSA.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/MDP/Environments/CliffProblem.hpp>

BOOST_AUTO_TEST_CASE( cliff ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(12, 3);

    auto model = makeCliffProblem(grid);

    SARSA solver(model, 0.4);

    QGreedyPolicy gPolicy(solver.getQFunction());
    EpsilonPolicy ePolicy(gPolicy, 0.7);

    size_t start = model.getS() - 2;

    size_t s, a, a1;
    for ( int episode = 0; episode < 10000; ++episode ) {
        // Ok, so this is because I basically couldn't find
        // any other way to make this thing converge to the
        // solution pointed out in the Sutton & Barto book.
        // Aside from the fact that they say that the best
        // path is discovered in ~50 episodes, while here it's
        // not even remotely close, even then no way to keep
        // it stable unless the learning parameter is made
        // to converge.
        // No offense, but this to me screams "YAY QLEARNING"
        // Or it may be a bug in my implementation - if so
        // please let me know!
        if (episode > 500)
            solver.setLearningRate(5.0 / (episode/3 + 1));
        s = start;
        a = ePolicy.sampleAction( s );

        for ( int i = 0; i < 10000; ++i ) {
            const auto [s1, rew] = model.sampleSR( s, a );
            a1 = ePolicy.sampleAction( s );
            solver.stepUpdateQ( s, a, s1, a1, rew );
            s = s1; a = a1;
            if ( s1 == model.getS() - 1 ) break;
        }
    }

    // Final path should be: UPx3, RIGHTx11, DOWNx3. Total moves: 17
    // We can use states only from above the cliff though
    BOOST_CHECK_EQUAL( gPolicy.getActionProbability(start, UP), 1.0 );

    auto state = grid(0, 2);
    for ( int i = 0; i < 2; ++i ) {
        BOOST_CHECK_EQUAL( gPolicy.getActionProbability(state, UP), 1.0 );
        state = grid.getAdjacent(UP, state);
    }
    for ( int i = 0; i < 11; ++i ) {
        BOOST_CHECK_EQUAL( gPolicy.getActionProbability(state, RIGHT), 1.0 );
        state = grid.getAdjacent(RIGHT, state);
    }
    for ( int i = 0; i < 3; ++i ) {
        BOOST_CHECK_EQUAL( gPolicy.getActionProbability(state, DOWN), 1.0 );
        state = grid.getAdjacent(DOWN, state);
    }
}

BOOST_AUTO_TEST_CASE( exceptions ) {
    namespace mdp = AIToolbox::MDP;
    BOOST_CHECK_EXCEPTION(mdp::SARSA(1,1,0.0,0.5),   std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(mdp::SARSA(1,1,-10.0,0.5), std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(mdp::SARSA(1,1,3.0,0.5),   std::invalid_argument, [](const std::invalid_argument &){return true;});

    BOOST_CHECK_EXCEPTION(mdp::SARSA(1,1,0.3,0.0),   std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(mdp::SARSA(1,1,0.3,-0.5),  std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(mdp::SARSA(1,1,0.3,1.1),   std::invalid_argument, [](const std::invalid_argument &){return true;});
}
