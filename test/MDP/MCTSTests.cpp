#define BOOST_TEST_MODULE MDP_MCTS
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/MCTS.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include "CornerProblem.hpp"

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;

    auto model = makeCornerProblem();
    size_t S = model.getS(), A = model.getA();

    MCTS<decltype(model)> solver(model, 10000, 5.0);

    // Check that solution agrees with that we'd like
    //
    //   0,0
    //     +-------+-------+-------+-------+
    //     |   ^   |       |       |       |
    //     | <-+-> | <-+   | <-+   | <-+   |
    //     |   v   |       |       |   v   |
    //     +-------+-------+-------+-------+
    //     |   ^   |   ^   |   ^   |       |
    //     |   +   | <-+   | <-+-> |   +   |
    //     |       |       |   v   |   v   |
    //     +-------+-------+-------+-------+
    //     |   ^   |   ^   |       |       |
    //     |   +   | <-+-> |   +-> |   +   |
    //     |       |   v   |   v   |   v   |
    //     +-------+-------+-------+-------+
    //     |   ^   |       |       |   ^   |
    //     |   +-> |   +-> |   +-> | <-+-> |
    //     |       |       |       |   v   |
    //     +-------+-------+-------+-------+
    //                                     3,3

    // Self-absorbing states have all same values, so action does not matter.
    // Also cells in the diagonal are indifferent as to the chosen direction.

    // Middle top cells want to go left to the absorbing state:
    BOOST_CHECK_EQUAL( solver.sampleAction(1,10), State::LEFT);
    BOOST_CHECK_EQUAL( solver.sampleAction(2,10), State::LEFT);

    // Last cell of first row wants to either go down or left
    auto a = solver.sampleAction(3,10);
    BOOST_CHECK( a == State::LEFT || a == State::DOWN );

    // Middle cells of first column want to go up
    BOOST_CHECK_EQUAL( solver.sampleAction(4,10), State::UP);
    BOOST_CHECK_EQUAL( solver.sampleAction(8,10), State::UP);

    // Cell in 1,1 wants left + up
    a = solver.sampleAction(5,10);
    BOOST_CHECK( a == State::LEFT || a == State::UP );

    // Middle cells of last column want to go down
    BOOST_CHECK_EQUAL( solver.sampleAction(7, 10), State::DOWN);
    BOOST_CHECK_EQUAL( solver.sampleAction(11,10), State::DOWN);

    // Cell in 2,2 wants right + down
    a = solver.sampleAction(10,10);
    BOOST_CHECK( a == State::RIGHT || a == State::DOWN );

    // Bottom cell in first column wants up + right
    a = solver.sampleAction(12,10);
    BOOST_CHECK( a == State::RIGHT || a == State::UP );

    // Finally, bottom middle cells want just right
    BOOST_CHECK_EQUAL( solver.sampleAction(13,10), State::RIGHT);
    BOOST_CHECK_EQUAL( solver.sampleAction(14,10), State::RIGHT);
}
