#ifndef AI_TOOLBOX_MDP_CLIFF_PROBLEM
#define AI_TOOLBOX_MDP_CLIFF_PROBLEM

#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

namespace AIToolbox::MDP {

    /**
     * @brief This function sets up the cliff problem in a SparseModel.
     *
     * The gist of this problem is a small grid where the agent is suppose to walk
     * from a state to another state. The only problem is that between the two
     * points stands a cliff, and walking down the cliff results in a huge negative
     * reward, and in the agent being reset at the start of the walk. Reaching the
     * end results in a positive reward, while every step results in a small
     * negative reward.
     *
     * Movement here is fully deterministic.
     *
     *  +--------+--------+--------+--------+--------+
     *  |        |        |        |        |        |
     *  |    0   |    1   |  ....  |   X-2  |   X-1  |
     *  |        |        |        |        |        |
     *  +--------+--------+--------+--------+--------+
     *  |        |        |        |        |        |
     *  |    X   |   X+1  |  ....  |  2X-2  |  2X-1  |
     *  |        |        |        |        |        |
     *  +--------+--------+--------+--------+--------+
     *  |        |        |        |        |        |
     *  |   2X   |  2X+1  |  ....  |  3X-2  |  3X-1  |
     *  |        |        |        |        |        |
     *  +--------+--------+--------+--------+--------+
     *  |        |        |        |        |        |
     *  |  ....  |  ....  |  ....  |  ....  |  ....  |
     *  |        |        |        |        |        |
     *  +--------+--------+--------+--------+--------+
     *  |        |        |        |        |        |
     *  | (Y-1)X |(Y-1)X+1|  ....  |  YX-2  |  YX-1  |
     *  |        |        |        |        |        |
     *  +--------+--------+--------+--------+--------+
     *  | (START)|        |        |        | (GOAL) |
     *  |   YX   |  ~~~~  |  ....  |  ~~~~  |  YX+1  |
     *  |        |        |        |        |        |
     *  +--------+--------+--------+--------+--------+
     *               \                 /
     *                --------- -------
     *                         V
     *                     The Cliff
     *
     * To do this we use a grid above the cliff, and we attach two
     * states under it.
     *
     * @param grid The grid to use for the problem.
     *
     * @return The SparseModel representing the problem.
     */
    inline AIToolbox::MDP::SparseModel makeCliffProblem(const GridWorld & grid) {
        using namespace GridWorldUtils;

        size_t S = grid.getWidth() * grid.getHeight() + 2;
        size_t A = 4;

        AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
        AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);

        double failReward = -100.0, stepReward = -1.0, winReward = 0.0;

        // Default all transitions within the grid to be deterministic,
        // and give negative reward. Remember that the actual cliff is
        // under the grid.
        for ( size_t s = 0; s < S-2; ++s ) {
            for ( size_t a = 0; a < A; ++a ) {
                auto cell = grid.getAdjacent((Direction)a, grid(s));
                transitions[s][a][cell] = 1.0;
                rewards[s][a][cell] = stepReward;
            }
        }
        // Attach start and goal states
        size_t start = S - 2, goal = S - 1;
        size_t upStart = (grid.getHeight() - 1) * grid.getWidth();
        size_t upGoal  = S - 3;

        // Fix start
        transitions[start][UP   ][upStart] =  1.0;
        rewards    [start][UP   ][upStart] = stepReward;
        transitions[start][LEFT ][start  ] =  1.0;
        rewards    [start][LEFT ][start  ] = stepReward;
        transitions[start][DOWN ][start  ] =  1.0;
        rewards    [start][DOWN ][start  ] = stepReward;
        transitions[start][RIGHT][start  ] =  1.0;
        rewards    [start][RIGHT][start  ] = failReward; // This goes into the cliff

        // Fix down for upStart
        transitions[upStart][DOWN][upStart] = 0.0;
        rewards    [upStart][DOWN][upStart] = 0.0;
        transitions[upStart][DOWN][start  ] = 1.0;
        rewards    [upStart][DOWN][start  ] = stepReward;

        // Fix goal (self absorbing)
        transitions[goal][UP   ][goal] = 1.0;
        transitions[goal][LEFT ][goal] = 1.0;
        transitions[goal][DOWN ][goal] = 1.0;
        transitions[goal][RIGHT][goal] = 1.0;

        // Fix upGoal
        transitions[upGoal][DOWN][upGoal] = 0.0;
        rewards    [upGoal][DOWN][upGoal] = 0.0;
        transitions[upGoal][DOWN][goal  ] = 1.0;
        rewards    [upGoal][DOWN][goal  ] = winReward; // Won!

        // Fix cliff edge
        for ( size_t s = upStart + 1; s < upGoal; ++s ) {
            transitions[s][DOWN][s    ] = 0.0;
            rewards    [s][DOWN][s    ] = 0.0;
            transitions[s][DOWN][start] = 1.0;
            rewards    [s][DOWN][start] = failReward; // This goes into the cliff
        }

        SparseModel model(S, A, transitions, rewards, 1.0);

        return model;
    }
}

#endif
