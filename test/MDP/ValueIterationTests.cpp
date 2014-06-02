#define BOOST_TEST_MODULE MDP_ValueIteration
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/Model.hpp>

// The gist of this problem is a small grid where
// the upper-left corner and the bottom-right corner
// are self-absorbing states. The agent can move in a
// top-left-down-right way, where each transition that
// is not self absorbing results in a reward penalty of -1.
// In addition the movements are not guaranteed: the
// agent succeeds only 80% of the time.
//
// Thus the agent needs to be able to find the shortest
// path to one of the self-absorbing states from every other
// state.

// Here we define an helper class that allows us to easily
// convert between our map coordinates to unique states, to
// use easily the library.
class State {
    public:
        static constexpr int MAP_SIZE = 4;
        State() : x_(0), y_(0) {}
        State(int x, int y) { setX(x); setY(y); }
        State(size_t s) : x_(s%MAP_SIZE), y_(s/MAP_SIZE) {}

        operator size_t() { return x_ + y_*MAP_SIZE; }

        enum Direction { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

        void adjacent(Direction d) {
            switch ( d ) {
                case UP:    setY(y_-1); return;
                case DOWN:  setY(y_+1); return;
                case LEFT:  setX(x_-1); return;
                case RIGHT: setX(x_+1); return;
            }
        }

        void setX(int newX) {
            if ( newX < 0 ) x_ = 0;
            else if ( newX >= MAP_SIZE ) x_ = MAP_SIZE - 1;
            else x_ = newX;
        }
        void setY(int newY) {
            if ( newY < 0 ) y_ = 0;
            else if ( newY >= MAP_SIZE ) y_ = MAP_SIZE - 1;
            else y_ = newY;
        }
        int getX() const { return x_; }
        int getY() const { return y_; }
    private:
        int x_, y_;
};

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;

    size_t S = 16, A = 4;

    Model::TransitionTable transitions(boost::extents[S][A][S]);
    Model::RewardTable rewards(boost::extents[S][A][S]);

    for ( int x = 0; x < 4; ++x ) {
        for ( int y = 0; y < 4; ++y ) {
            State s(x,y);
            if ( s == 0 || s == 15 ) {
                // Self absorbing states
                for ( size_t a = 0; a < A; ++a )
                    transitions[s][a][s] = 1.0;
            }
            else {
                for ( size_t a = 0; a < A; ++a ) {
                    State s1 = s;
                    s1.adjacent((State::Direction)a);
                    // If the move takes you outside the map, it doesn't do
                    // anything
                    if ( s == s1 ) transitions[s][a][s1] = 1.0;
                    else {
                        transitions[s][a][s1] = 0.8;
                        transitions[s][a][s] = 0.2;
                    }
                    rewards[s][a][s1] = -1.0;
                }
            }
        }
    }

    Model model(S, A, transitions, rewards, 1.0);
    ValueIteration solver;

    auto solution = solver(model);
    // Check that problem has been solved
    BOOST_CHECK( std::get<0>(solution) );
    // Get best policy from QFunction
    auto & qfun = std::get<2>(solution);
    QGreedyPolicy policy( qfun );

    // Check that solution agrees with that we'd like
    //
    //  +-------+-------+-------+-------+
    //  |   ^   |       |       |       |
    //  | <-+-> | <-+   | <-+   | <-+   |
    //  |   v   |       |       |   v   |
    //  +-------+-------+-------+-------+
    //  |   ^   |   ^   |   ^   |       |
    //  |   +   | <-+   | <-+-> |   +   |
    //  |       |       |   v   |   v   |
    //  +-------+-------+-------+-------+
    //  |   ^   |   ^   |       |       |
    //  |   +   | <-+-> |   +-> |   +   |
    //  |       |   v   |   v   |   v   |
    //  +-------+-------+-------+-------+
    //  |   ^   |       |       |   ^   |
    //  |   +-> |   +-> |   +-> | <-+-> |
    //  |       |       |       |   v   |
    //  +-------+-------+-------+-------+

    // Self-absorbing states have all same values, so action does not matter.
    // Also cells in the diagonal are indifferent as to the chosen direction.
    for ( size_t a = 0; a < A; ++a ) {
        BOOST_CHECK_EQUAL( policy.getActionProbability(0, a),   0.25);
        BOOST_CHECK_EQUAL( policy.getActionProbability(6, a),   0.25);
        BOOST_CHECK_EQUAL( policy.getActionProbability(9, a),   0.25);
        BOOST_CHECK_EQUAL( policy.getActionProbability(15, a),  0.25);
    }

    // Middle top cells want to go left to the absorbing state:
    BOOST_CHECK_EQUAL( policy.getActionProbability(1, State::LEFT), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(2, State::LEFT), 1.0);

    // Last cell of first row wants to either go down or left
    BOOST_CHECK_EQUAL( policy.getActionProbability(3, State::LEFT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(3, State::DOWN), 0.5);

    // Middle cells of first column want to go up
    BOOST_CHECK_EQUAL( policy.getActionProbability(4, State::UP), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(8, State::UP), 1.0);

    // Cell in 1,1 wants left + up
    BOOST_CHECK_EQUAL( policy.getActionProbability(5, State::LEFT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(5, State::UP), 0.5);

    // Middle cells of last column want to go down
    BOOST_CHECK_EQUAL( policy.getActionProbability(7,  State::DOWN), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(11, State::DOWN), 1.0);

    // Cell in 2,2 wants right + down
    BOOST_CHECK_EQUAL( policy.getActionProbability(10, State::RIGHT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(10, State::DOWN), 0.5);

    // Bottom cell in first column wants up + right
    BOOST_CHECK_EQUAL( policy.getActionProbability(12, State::RIGHT), 0.5);
    BOOST_CHECK_EQUAL( policy.getActionProbability(12, State::UP), 0.5);

    // Finally, bottom middle cells want just right
    BOOST_CHECK_EQUAL( policy.getActionProbability(13, State::RIGHT), 1.0);
    BOOST_CHECK_EQUAL( policy.getActionProbability(14, State::RIGHT), 1.0);

    // Verify that ValueFunction holds the correct actions.
    auto & vfun = std::get<1>(solution);
    auto & values = std::get<VALUES>(vfun);
    auto & actions = std::get<ACTIONS>(vfun);
    for ( size_t s = 0; s < S; ++s ) {
        // We check that values correspond between Q and V
        BOOST_CHECK_EQUAL( qfun[s][actions[s]], values[s] );

        // And that the action truly points to (one of) the best.
        auto ref = qfun[s];
        auto maxIt = std::max_element(std::begin(ref), std::end(ref));
        BOOST_CHECK_EQUAL( *maxIt, values[s] );
    }
}
