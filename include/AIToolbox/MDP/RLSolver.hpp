#ifndef AI_TOOLBOX_MDP_RLSOLVER_HEADER_FILE
#define AI_TOOLBOX_MDP_RLSOLVER_HEADER_FILE

namespace AIToolbox {
    namespace MDP {
        class Solution;
        class RLModel;
        class Experience;

        /**
         * @brief This class is an interface for all Reinforcement Learning MDP solvers.
         */
        class RLSolver {
            public:
                /**
                 * @brief This function solves an MDP RLModel.
                 *
                 * Ideally this is the main function where the
                 * solving process happens. Different methods may
                 * need to implement auxiliary functions to help
                 * them work.
                 * 
                 * It is important to notice that, differently from the
                 * Solver::operator(), this function IS allowed to modify
                 * both the underlying Experience of the RLModel and the
                 * RLModel itself. This can't be avoided as RLModels are by
                 * design learned, and so they need to be refined to approximate
                 * the true MDP, while a Model represents a true defined MDP and
                 * thus there is no reason for the solution method to change it.
                 *
                 * @param e The underlying experience of the RLModel that needs to be solved.
                 * @param m The RLModel that needs to be solved.
                 * @param s The Solution that is going to be outputted.
                 *
                 * @return True if the solving process succeeded, false otherwise.
                 */
                virtual bool operator()(Experience & e, RLModel & m, Solution & s) = 0;
        };
    }
}

#endif
