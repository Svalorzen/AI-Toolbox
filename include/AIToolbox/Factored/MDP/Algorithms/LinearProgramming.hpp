#ifndef AI_TOOLBOX_FACTORED_MDP_LINEAR_PROGRAMMING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_LINEAR_PROGRAMMING_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox { class LP; }

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class solves a factored MDP with Linear Programming.
     *
     * This class computes best approximation possible of the optimal
     * ValueFunction with respect to the input basis functions.
     *
     * The process is very similar to the one performed by
     * AIToolbox::MDP::LinearProgramming. However, since we can't create
     * constraints for every possible state action pair here (for obvious space
     * reasons), we use the mechanism introduced in FactoredLP: we build a
     * series of constraints using VariableElimination that are equivalent to
     * the exponential constraints, while being actually linear in the number
     * of basis functions.
     *
     * This results in a method that can very approximate very well the optimal
     * ValueFunction for environments with trillion or more states and actions,
     * in a reasonable amount of time.
     */
    class LinearProgramming {
        public:
            /**
             * @brief This function solves the input MDP using linear programming.
             *
             * @param m The MDP that needs to be solved.
             * @param h The basis functions to use to approximate V*.
             *
             * @return A tuple containing the weights for the basis functions, and the equivalent QFunction.
             */
            std::tuple<Vector, QFunction> operator()(const CooperativeModel & m, const FactoredVector & h) const;

        private:
            using Rule = std::pair<PartialValues, size_t>;
            using Rules = std::vector<Rule>;
            using Graph = FactorGraph<Rules>;

            /**
             * @brief This function sets up and solves the underlying LP.
             *
             * @param m The model to solve.
             * @param g A precomputed backpropagation of the basis functions.
             * @param h The basis functions.
             *
             * @return The output of the LP solving process.
             */
            std::optional<Vector> solveLP(const CooperativeModel & m, const FactoredMatrix2D & g, const FactoredVector & h) const;

            /**
             * @brief This function performs a step in the variable elimination process.
             *
             * As it removes the state, this function will add to the input LP
             * all rules needed to represent the variable elimination process.
             *
             * @param F The factor space, of states and actions combined.
             * @param graph The graph to perform VE on.
             * @param s The factor to remove.
             * @param lp The LP to modify.
             * @param finalFactors The variable where to store the final rules' ids.
             */
            void removeState(const Factors & F, Graph & graph, size_t s, LP & lp, std::vector<size_t> & finalFactors) const;
    };
}

#endif
