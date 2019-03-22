#define BOOST_TEST_MODULE MDP_QLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Algorithms/QLearning.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/MDP/Environments/CliffProblem.hpp>

BOOST_AUTO_TEST_CASE( updates ) {
    using namespace AIToolbox::MDP;

    QLearning solver(5, 5, 0.9, 0.5);
    {
        // State goes to itself, thus needs to consider
        // next-step value.
        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 0), 5.0 );

        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 0), 9.75 );
    }
    {
        // Here it does not, so improvement is slower.
        solver.stepUpdateQ(3, 0, 4, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(3, 0), 5.0 );

        solver.stepUpdateQ(3, 0, 4, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(3, 0), 7.50 );
    }
    {
        // Test that index combinations are right.
        solver.stepUpdateQ(0, 1, 1, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 1), 5.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()(1, 0), 0.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()(1, 1), 0.0  );
    }
}

BOOST_AUTO_TEST_CASE( cliff ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(12, 3);

    auto model = makeCliffProblem(grid);

    QLearning solver(model, 0.5);

    QGreedyPolicy gPolicy(solver.getQFunction());
    EpsilonPolicy ePolicy(gPolicy, 0.1);

    size_t start = model.getS() - 2;

    size_t s, a;
    for ( int episode = 0; episode < 100; ++episode ) {
        s = start;
        for ( int i = 0; i < 10000; ++i ) {
            a = ePolicy.sampleAction( s );
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

BOOST_AUTO_TEST_CASE( exceptions ) {
    using namespace AIToolbox::MDP;

    BOOST_CHECK_EXCEPTION(QLearning(1,1,0.0,0.5),   std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(QLearning(1,1,-10.0,0.5), std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(QLearning(1,1,3.0,0.5),   std::invalid_argument, [](const std::invalid_argument &){return true;});

    BOOST_CHECK_EXCEPTION(QLearning(1,1,0.3,0.0),   std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(QLearning(1,1,0.3,-0.5),  std::invalid_argument, [](const std::invalid_argument &){return true;});
    BOOST_CHECK_EXCEPTION(QLearning(1,1,0.3,1.1),   std::invalid_argument, [](const std::invalid_argument &){return true;});
}
