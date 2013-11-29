#ifndef AI_TOOLBOX_MDP_SOLVER_HEADER_FILE
#define AI_TOOLBOX_MDP_SOLVER_HEADER_FILE

namespace AIToolbox {
    namespace MDP {
        class Solution;
        class Model;

        /**
         * @brief This class is an interface for all MDP solvers.
         */
        class Solver {
            public:
                /**
                 * @brief This function solves an MDP Model.
                 *
                 * Ideally this is the main function where the
                 * solving process happens. Different methods may
                 * need to implement auxiliary functions to help
                 * them work.
                 *
                 * @param m The Model that needs to be solved.
                 * @param s The Solution that is going to be outputted.
                 *
                 * @return True if the solving process succeeded, false otherwise.
                 */
                virtual bool operator()(const Model & m, Solution & s) = 0;
        };
    }
}

#endif
