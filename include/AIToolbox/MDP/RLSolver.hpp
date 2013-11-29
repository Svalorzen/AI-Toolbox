#ifndef AI_TOOLBOX_MDP_RLSOLVER_HEADER_FILE
#define AI_TOOLBOX_MDP_RLSOLVER_HEADER_FILE

namespace AIToolbox {
    namespace MDP {
        class Solution;
        class RLModel;
        class Experience;

        class RLSolver {
            public:
                virtual bool operator()(Experience &, RLModel &, Solution & s) = 0;
        };
    }
}

#endif
