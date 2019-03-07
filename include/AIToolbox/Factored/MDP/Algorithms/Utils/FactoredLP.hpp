#ifndef AI_TOOLBOX_FACTORED_MDP_FACTORED_LP_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_FACTORED_LP_HEADER_FILE

#include <utility>

#include <AIToolbox/Factored/MDP/Types.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class represents the Factored LP algorithm.
     *
     * This algorithm has been introduced in a number of Guestrin et al.
     * papers. The Factored LP algorithm takes part in approximately solving
     * factored state MDPs.
     *
     * The idea is that the Value Function for such a factored MDP is
     * approximated through a series of basis functions, which are chosen by
     * hand by the user. These functions are linearly combined in order to
     * produce as close an approximation to the real Value Function as
     * possible.
     *
     * This allows to limit the complexity of the Value Function when, for
     * example, iterating through the steps of Value Iteration.
     *
     * Note that the input Value Function in this algorithm should most likely
     * have been produced by some step which has made it not a linear sum of
     * the basis functions.
     *
     * This algorithm is thus used to find the coefficients that have to be
     * applied to the basis functions in order to approximate the input Value
     * Function. Once that's done, the basis functions can be summed and the
     * approximate Value Function constructed in order to continue whatever
     * algorithm is being executed.
     */
    class FactoredLP {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor is used to initialized the graph that contains
             * references to all the rules built within the underlying LP.
             *
             * @param s The state space of the problem.
             */
            FactoredLP(State s) : S(std::move(s)) {}

            /**
             * @brief This function finds the coefficients to approximate a Value Function.
             *
             * Differently from VariableElimination, we take our inputs already
             * in the form of a graph. This avoids us a bit of work here since
             * we'd have to build the graphs anyway in order to correctly
             * process the inputs.
             *
             * Since the main task of this class is to setup and run an LP, we
             * return its result as-is, without checking if the LP succeeded or
             * failed. We don't know enough here to be sure of what the
             * algorithm calling us wants to do, so we defer responsibility to
             * it.
             *
             * This function allows to optionally request the usage of a
             * constant basis for C. A constant basis has a value of 1 for
             * every possible state. We don't want to a constant basis
             * explicitly to C as (1) VE won't work and (2) it requires
             * specifying an explicit value of 1 for every possible state,
             * which is infeasible. If a constant basis is requested, the
             * return value will contain an additional coefficient at the end
             * for the constant basis.
             *
             * @param C The basis functions used to approximate the Value Function.
             * @param b The Value Function to approximate.
             * @param addConstantBasis Whether we should include an impled constant basis for C.
             *
             * @return The coefficients used to linearly combine the basis functions.
             */
            std::optional<Vector> operator()(const FactoredVector & C, const FactoredVector & b, bool addConstantBasis = false);

        private:
            State S;
    };
}

#endif
