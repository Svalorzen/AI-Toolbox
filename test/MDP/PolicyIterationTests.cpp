#define BOOST_TEST_MODULE MDP_PolicyIteration
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/PolicyIteration.hpp>

#include <type_traits>

#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

#include "Utils/OldMDPModel.hpp"

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(4, 4);

    Model model = makeCornerProblem(grid);
    size_t A = model.getA();

    // We set the horizon to a very high value so that
    // the tolerance bound will prevail, solving the problem
    // until convergence (infinite horizon).
    PolicyIteration solver(1000000, 0.001);

    auto qfun = solver(model);
    QGreedyPolicy policy( qfun );

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
    for ( size_t a = 0; a < A; ++a ) {
        BOOST_CHECK_EQUAL( policy.getActionProbability(0, a),   0.25);
        BOOST_CHECK_EQUAL( policy.getActionProbability(6, a),   0.25);
        BOOST_CHECK_EQUAL( policy.getActionProbability(9, a),   0.25);
        BOOST_CHECK_EQUAL( policy.getActionProbability(15, a),  0.25);
    }

    // Middle top cells want to go left to the absorbing state:
    BOOST_CHECK_EQUAL( policy.getActionProbability(1, LEFT), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(2, LEFT), 1.0);

    // Last cell of first row wants to either go down or left
    BOOST_CHECK_EQUAL( policy.getActionProbability(3, LEFT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(3, DOWN), 0.5);

    // Middle cells of first column want to go up
    BOOST_CHECK_EQUAL( policy.getActionProbability(4, UP), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(8, UP), 1.0);

    // Cell in 1,1 wants left + up
    BOOST_CHECK_EQUAL( policy.getActionProbability(5, LEFT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(5, UP),   0.5);

    // Middle cells of last column want to go down
    BOOST_CHECK_EQUAL( policy.getActionProbability(7,  DOWN), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(11, DOWN), 1.0);

    // Cell in 2,2 wants right + down
    BOOST_CHECK_EQUAL( policy.getActionProbability(10, RIGHT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(10, DOWN),  0.5);

    // Bottom cell in first column wants up + right
    BOOST_CHECK_EQUAL( policy.getActionProbability(12, RIGHT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(12, UP),    0.5);

    // Finally, bottom middle cells want just right
    BOOST_CHECK_EQUAL( policy.getActionProbability(13, RIGHT), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(14, RIGHT), 1.0);
}
