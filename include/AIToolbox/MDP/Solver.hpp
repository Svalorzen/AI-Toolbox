#ifndef AI_TOOLBOX_MDP_SOLVER_HEADER_FILE
#define AI_TOOLBOX_MDP_SOLVER_HEADER_FILE

#include <tuple>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
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
                 * solving process happens. Different algorithms may
                 * need to implement auxiliary functions to help
                 * them work.
                 * 
                 * Users of this class should check whether the final
                 * Solution is valid or invalid.
                 *
                 * @param m The Model that needs to be solved.
                 * @return A tuple containing a boolean value specifying the
                 *         return status of the solution problem, the
                 *         ValueFunction and the QFunction for the Model.
                 */
                virtual std::tuple<bool, ValueFunction, QFunction> operator()(const Model & m) = 0;
        };
    }
}

#endif
