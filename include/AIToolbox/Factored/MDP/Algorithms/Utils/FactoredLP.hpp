#ifndef AI_TOOLBOX_FACTORED_MDP_FACTORED_LP_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_FACTORED_LP_HEADER_FILE

#include <utility>

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Utils/Test.hpp>

namespace AIToolbox { class LP; }

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
             * @param C The basis functions used to approximate the Value Function.
             * @param b The Value Function to approximate.
             *
             * @return The coefficients used to linearly combine the basis functions.
             */
            std::optional<Vector> operator()(const FactoredFunction<1> & C, const FactoredFunction<1> & b);

        private:
            using Rule = std::tuple<PartialState, size_t>;
            using Rules = std::vector<Rule>;
            using Graph = FactorGraph<Rules>;

            /**
             * @brief This function performs a step in the variable elimination process.
             *
             * As it removes the state, this function will add to the input LP
             * all rules needed to represent the variable elimination process.
             *
             * @param graph The graph to perform VE on.
             * @param s The factor to remove.
             * @param lp The LP to modify.
             * @param finalFactors The variable where to store the final rules' ids.
             */
            void removeState(Graph & graph, size_t s, LP & lp, std::vector<size_t> & finalFactors);

            State S;
    };
}

#endif
