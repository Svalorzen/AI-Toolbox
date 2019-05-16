#define BOOST_TEST_MODULE MDP_MCTS_EXTENDED
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/dynamic_bitset.hpp>
#include <iostream>

#include <AIToolbox/MDP/Algorithms/MCTS.hpp>
#include <AIToolbox/MDP/Algorithms/UCB.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

class BitsetModel : public AIToolbox::MDP::Model {
    public:
        BitsetModel(const AIToolbox::MDP::Model &m) : AIToolbox::MDP::Model(m), model_(m) {}

        std::tuple<size_t, double> sampleSR(const size_t s, const boost::dynamic_bitset<> a) const {
            return model_.sampleSR(s, a.to_ulong());
        }

        std::vector<boost::dynamic_bitset<>> getAllowedActions(const size_t) const {
            std::vector<boost::dynamic_bitset<>> v;
            for(size_t i = 0; i < model_.getA(); i++)
                v.push_back(boost::dynamic_bitset<>(2, i));
            return v;
        }
    private:
      const AIToolbox::MDP::Model &model_;
};

class ExtendedUCB : public AIToolbox::MDP::UCB {
    public:
        void initializeActions(AIToolbox::MDP::MCTS<BitsetModel, ExtendedUCB, size_t, boost::dynamic_bitset<>>::StateNode &parent, const BitsetModel &m) const {
            if (parent.children.size() == 0) {
                size_t A = m.getA();
                parent.children.resize(A);
                for (size_t i = 0; i < A; i++) {
                    parent.children.at(i).action = boost::dynamic_bitset(2, i);
                }
            }
        }
};

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;
    using namespace GridWorldEnums;

    GridWorld grid(4,4);

    auto m = makeCornerProblem(grid);
    BitsetModel model(m);

    ExtendedUCB ucb;
    MCTS<BitsetModel, ExtendedUCB, size_t, boost::dynamic_bitset<>> solver(model, ucb, 10000, 5.0);

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

    auto m = makeCornerProblem(grid);
    BitsetModel model(m);

    ExtendedUCB ucb;
    MCTS<BitsetModel, ExtendedUCB, size_t, boost::dynamic_bitset<>> solver(model, ucb, 1, 5.0);

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
