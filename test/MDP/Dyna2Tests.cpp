#define BOOST_TEST_MODULE MDP_DYNA2
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/Dyna2.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/Policies/EpsilonPolicy.hpp>

#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(4,4);

    auto model = makeCornerProblem(grid);

    Dyna2 solver(model, 0.1, 0.9, 0.001, 50);

    // This policy we use just to check that we have learned to solve the model
    // in the permanent learning.
    QGreedyPolicy p1(solver.getPermanentQFunction());

    // This is the policy we use to act, based on the transient learning.
    QGreedyPolicy p2(solver.getTransientQFunction());
    EpsilonPolicy p3(p2);

    // This is the policy we use during batch updates (owned by the solver).
    solver.setInternalPolicy(new EpsilonPolicy(p2, 0.4));

    AIToolbox::RandomEngine rand(AIToolbox::Impl::Seeder::getSeed());
    std::uniform_int_distribution<size_t> initState(0, model.getS() - 1);

    for (size_t e = 0; e < 3000; ++e) {
        solver.resetTransientLearning();

        size_t s = initState(rand);
        auto a = p3.sampleAction(s);

        for (size_t t = 0; t < 100; ++t) {
            const auto [s1, r] = model.sampleSR(s, a);
            const auto a1 = p3.sampleAction(s);

            solver.stepUpdateQ(s, a, s1, a1, r);
            solver.batchUpdateQ(s1);

            if (model.isTerminal(s1)) break;
            s = s1;
            a = a1;
        }
    }

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
    BOOST_CHECK_EQUAL( p1.sampleAction(1), LEFT);
    BOOST_CHECK_EQUAL( p1.sampleAction(2), LEFT);

    // Last cell of first row wants to either go down or left
    auto a = p1.sampleAction(3);
    BOOST_CHECK( a == LEFT || a == DOWN );

    // Middle cells of first column want to go up
    BOOST_CHECK_EQUAL( p1.sampleAction(4), UP);
    BOOST_CHECK_EQUAL( p1.sampleAction(8), UP);

    // Cell in 1,1 wants left + up
    a = p1.sampleAction(5);
    BOOST_CHECK( a == LEFT || a == UP );

    // Middle cells of last column want to go down
    BOOST_CHECK_EQUAL( p1.sampleAction(7), DOWN);
    BOOST_CHECK_EQUAL( p1.sampleAction(11), DOWN);

    // Cell in 2,2 wants right + down
    a = p1.sampleAction(10);
    BOOST_CHECK( a == RIGHT || a == DOWN );

    // Bottom cell in first column wants up + right
    a = p1.sampleAction(12);
    BOOST_CHECK( a == RIGHT || a == UP );

    // Finally, bottom middle cells want just right
    BOOST_CHECK_EQUAL( p1.sampleAction(13), RIGHT);
    BOOST_CHECK_EQUAL( p1.sampleAction(14), RIGHT);
}
