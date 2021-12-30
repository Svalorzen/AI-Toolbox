#ifndef AI_TOOLBOX_FACTORED_BANDIT_REUSING_ITERATIVE_LOCAL_SEARCH_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_REUSING_ITERATIVE_LOCAL_SEARCH_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class approximately finds the best joint action with Reusing Iterative Local Search.
     *
     * This class is mostly a wrapper around LocalSearch. The idea is to avoid
     * local optima by adding noise to the solution found by LocalSearch, or
     * alternatively restart from random points to see whether we can find a
     * better solution.
     *
     * In addition, we cache the best action found, so that we can re-use it as
     * a starting point if needed. The idea is that if the graph to solve has
     * changed in a relatively minor way, it is likely that the optimal
     * solution will be close to the one found previously. Note that this
     * caching is optional in case it is known that the graph changed
     * substantially (or we want to solve a different graph).
     */
    class ReusingIterativeLocalSearch {
        public:
            using Result = std::tuple<Action, double>;
            using Graph = LocalSearch::Graph;

            /**
             * @brief Basic constructor.
             */
            ReusingIterativeLocalSearch(double resetActionProbability, double randomizeFactorProbability, unsigned trialNum);

            /**
             * @brief This function approximately finds the best Action-value pair for the provided QFunctionRules.
             *
             * This function automatically sets up the Graph to perform local
             * search on from an iterable of QFunctionRules.
             *
             * On first call, this function optimizes over a single randomly
             * sampled initial action. Subsequently it will optimize using the
             * last best action as a starting point, unless it is explicitly
             * reset.
             *
             * \sa operator()(const Action &, Graph &, bool)
             *
             * @param A The action space of the agents.
             * @param rules An iterable object over QFunctionRules.
             * @param resetAction Whether to reset the internally cached last best action.
             *
             * @return A tuple containing the best Action and its value over the input rules.
             */
            template <typename Iterable>
            Result operator()(const Action & A, const Iterable & inputRules, bool resetAction = false) {
                auto graph = LocalSearch::makeGraph(A, inputRules);

                return (*this)(A, graph, resetAction);
            }

            /**
             * @brief This function approximately finds the best Action-value pair for the provided Graph.
             *
             * On first call, this function optimizes over a single randomly
             * sampled initial action. Subsequently it will optimize using the
             * last best action as a starting point, unless it is explicitly
             * reset.
             *
             * We perform
             *
             * \sa operator()(const Action &, Graph &, bool)
             *
             * @param A The action space of the agents.
             * @param rules An iterable object over QFunctionRules.
             * @param resetAction Whether to reset the internally cached last best action.
             *
             * @return A tuple containing the best Action and its value over the input rules.
             */
            Result operator()(const Action & A, const Graph & graph, bool resetAction = false);

        private:
            // Parameters
            double resetActionProbability_;
            double randomizeFactorProbability_;
            unsigned trialNum_;

            // Caches
            Action action_, newAction_;
            double value_;

            // Nested local search
            LocalSearch ls_;

            RandomEngine rnd_;
    };
};

#endif
