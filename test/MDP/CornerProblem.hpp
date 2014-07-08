#ifndef AI_TOOLBOX_MDP_CORNER_PROBLEM
#define AI_TOOLBOX_MDP_CORNER_PROBLEM

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
// 
// The grid cells are numbered as following:
//
//
//  +-------+-------+-------+-------+
//  |       |       |       |       |
//  |   0   |   1   |   2   |   3   |
//  |       |       |       |       |
//  +-------+-------+-------+-------+
//  |       |       |       |       |
//  |   4   |   5   |   6   |   7   |
//  |       |       |       |       |
//  +-------+-------+-------+-------+
//  |       |       |       |       |
//  |   8   |   9   |   10  |   11  |
//  |       |       |       |       |
//  +-------+-------+-------+-------+
//  |       |       |       |       |
//  |   12  |   13  |   14  |   15  |
//  |       |       |       |       |
//  +-------+-------+-------+-------+

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

inline AIToolbox::MDP::Model makeCornerProblem() {
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

    return model;
}

#endif
