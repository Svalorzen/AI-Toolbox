#ifndef AI_TOOLBOX_MDP_SOLVER_HEADER_FILE
#define AI_TOOLBOX_MDP_SOLVER_HEADER_FILE

namespace AIToolbox {
    namespace MDP {
        class Solution;
        class Model;

        class Solver {
            public:
                virtual bool solve(const Model &, Solution & s) = 0;
        };
    }
}

#endif
