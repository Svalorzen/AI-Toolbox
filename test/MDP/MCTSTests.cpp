#define BOOST_TEST_MODULE MDP_MCTS
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "GlobalFixtures.hpp"

#include <AIToolbox/MDP/Algorithms/MCTS.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldUtils;

    GridWorld grid(4,4);

    auto model = makeCornerProblem(grid);

    MCTS solver(model, 10000, 5.0);

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
    BOOST_CHECK_EQUAL( solver.sampleAction(1,10), LEFT);
    BOOST_CHECK_EQUAL( solver.sampleAction(2,10), LEFT);

    // Last cell of first row wants to either go down or left
    auto a = solver.sampleAction(3,10);
    BOOST_CHECK( a == LEFT || a == DOWN );

    // Middle cells of first column want to go up
    BOOST_CHECK_EQUAL( solver.sampleAction(4,10), UP);
    BOOST_CHECK_EQUAL( solver.sampleAction(8,10), UP);

    // Cell in 1,1 wants left + up
    a = solver.sampleAction(5,10);
    BOOST_CHECK( a == LEFT || a == UP );

    // Middle cells of last column want to go down
    BOOST_CHECK_EQUAL( solver.sampleAction(7, 10), DOWN);
    BOOST_CHECK_EQUAL( solver.sampleAction(11,10), DOWN);

    // Cell in 2,2 wants right + down
    a = solver.sampleAction(10,10);
    BOOST_CHECK( a == RIGHT || a == DOWN );

    // Bottom cell in first column wants up + right
    a = solver.sampleAction(12,10);
    BOOST_CHECK( a == RIGHT || a == UP );

    // Finally, bottom middle cells want just right
    BOOST_CHECK_EQUAL( solver.sampleAction(13,10), RIGHT);
    BOOST_CHECK_EQUAL( solver.sampleAction(14,10), RIGHT);
}

BOOST_AUTO_TEST_CASE( sampleOneTime ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(4,4);

    auto model = makeCornerProblem(grid);

    MCTS solver(model, 1, 5.0);

    // We ensure MCTS does not crash when pruning a tree
    // and the new head was a leaf (and thus did not have
    // children).

    unsigned horizon = 2;

    // UCT here samples action 0, since it's
    // the first in line.
    solver.sampleAction(6, horizon);

    auto & graph_ = solver.getGraph();
    // We find the leaf we just produced
    auto it = graph_.children[0].children.begin();
    auto s1 = it->first;

    // We make a,o the new head
    solver.sampleAction( 0, s1, horizon - 1);
}

class VarActionModel {
    // This is a simple 3-door opening game where opening the middle door last
    // yields the highest reward.
    // - State = doors currently open (true = open, false = closed)
    // - Action = One of the closed doors.
    //   - 0 means "open the first closed door from the left"
    //   - 1 means "open the second closed door from the left"
    //   - 2 means "open the last closed door from the left"
    //   Number of actions is obviously equal to the number of closed doors left.
    public:
        using State = std::array<bool, 3>;

        State getS() const {
            return {};
        }
        size_t getA(const State & s) const {
            size_t A = 3;
            for (auto i = 0; i < 3; ++i)
                if (s[i]) --A;
            return A;
        }
        double getDiscount() const { return 0.9; }
        bool isTerminal(const State & s) const {
            for (auto b : s) if (!b) return false;
            return true;
        }
        std::tuple<State, double> sampleSR(const State & s, size_t a) const {
            State s1 = s;

            size_t count = 0, opened = 4;
            for (auto i = 0; i < 3; ++i) {
                if (s1[i]) ++count;
                else {
                    if (!a && opened == 4) {
                        opened = i;
                        s1[opened] = true;
                        ++count;
                    }
                    --a; // This will wrap around but it's ok
                }
            }
            double reward = 0.0;
            if (count == 3)
                reward = opened == 1 ? 5.0 : 1.0;
            return {s1, reward};
        }
};

BOOST_AUTO_TEST_CASE( variableActions ) {
    using namespace AIToolbox::MDP;

    VarActionModel model;
    model.sampleSR(model.getS(), model.getA(model.getS()));

    MCTS<decltype(model), boost::hash> solver(model, 100, 5.0);

    auto s = model.getS();

    // We simply act for 3 timesteps and check that indeed the last door we
    // leave open is the middle one, which gives the most reward (5.0).
    //
    // We also check that we recommend actions within the action space.
    auto a = solver.sampleAction(s, 3);
    BOOST_CHECK(a < model.getA(s));
    auto [s1, r1] = model.sampleSR(s, a);

    auto a1 = solver.sampleAction(a, s1, 2);
    BOOST_CHECK(a1 < model.getA(s1));
    auto [s2, r2] = model.sampleSR(s1, a1);

    auto a2 = solver.sampleAction(a1, s2, 1);
    BOOST_CHECK(a2 < model.getA(s2));
    auto [s3, r3] = model.sampleSR(s2, a2);

    // Just check that we behaved correctly
    BOOST_CHECK(r3 == 5.0);
}
