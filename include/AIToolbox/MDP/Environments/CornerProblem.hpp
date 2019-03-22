#ifndef AI_TOOLBOX_MDP_CORNER_PROBLEM
#define AI_TOOLBOX_MDP_CORNER_PROBLEM

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

namespace AIToolbox::MDP {
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
    //  +--------+--------+--------+--------+--------+
    //  | (GOAL) |        |        |        |        |
    //  |    0   |    1   |  ....  |   X-2  |   X-1  |
    //  |        |        |        |        |        |
    //  +--------+--------+--------+--------+--------+
    //  |        |        |        |        |        |
    //  |    X   |   X+1  |  ....  |  2X-2  |  2X-1  |
    //  |        |        |        |        |        |
    //  +--------+--------+--------+--------+--------+
    //  |        |        |        |        |        |
    //  |   2X   |  2X+1  |  ....  |  3X-2  |  3X-1  |
    //  |        |        |        |        |        |
    //  +--------+--------+--------+--------+--------+
    //  |        |        |        |        |        |
    //  |  ....  |  ....  |  ....  |  ....  |  ....  |
    //  |        |        |        |        |        |
    //  +--------+--------+--------+--------+--------+
    //  |        |        |        |        | (GOAL) |
    //  | (Y-1)X |(Y-1)X+1|  ....  |  YX-2  |  YX-1  |
    //  |        |        |        |        |        |
    //  +--------+--------+--------+--------+--------+

    inline AIToolbox::MDP::Model makeCornerProblem(const GridWorld & grid, double stepUncertainty = 0.8) {
        using namespace GridWorldEnums;

        size_t S = grid.getWidth() * grid.getHeight();
        size_t A = 4;

        AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
        AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);

        for ( int x = 0; x < 4; ++x ) {
            for ( int y = 0; y < 4; ++y ) {
                auto s = grid(x,y);
                if ( s == 0 || s == S-1 ) {
                    // Self absorbing states
                    for ( size_t a = 0; a < A; ++a )
                        transitions[s][a][s] = 1.0;
                }
                else {
                    for ( size_t a = 0; a < A; ++a ) {
                        auto s1 = grid.getAdjacent((Direction)a, s);
                        // If the move takes you outside the map, it doesn't do
                        // anything
                        if ( s == s1 ) transitions[s][a][s1] = 1.0;
                        else {
                            transitions[s][a][s1] = stepUncertainty;
                            transitions[s][a][s] = 1.0 - stepUncertainty;
                        }
                        rewards[s][a][s1] = -1.0;
                    }
                }
            }
        }
        return Model(S, A, transitions, rewards, 0.95);
    }
}

#endif
