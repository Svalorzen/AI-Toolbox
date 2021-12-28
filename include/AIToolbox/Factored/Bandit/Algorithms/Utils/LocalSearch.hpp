#ifndef AI_TOOLBOX_FACTORED_BANDIT_LOCAL_SEARCH_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_LOCAL_SEARCH_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class approximately finds the best joint action using Local Search.
     *
     * The Local Search algorithm is a simple routine that maximizes each agent
     * in turn, selecting its local action that maximizes the overall reward.
     *
     * We iteratively go over all agents (each time in random order to avoid
     * adversarial inputs), optimizing each one in turn, until no optimizations
     * can be done. In this way we are guaranteed to find a local optima, but
     * there is no guarantee that Local Search will be able to find the global
     * optima, which is why this is an approximate method.
     *
     * On the other hand, this method is quite fast, as each individual
     * optimization is simple and quick to do.
     */
    class LocalSearch {
        public:
            using Result = std::tuple<Action, double>;
            using Graph = FactorGraph<Vector>;

            /**
             * @brief This function builds the appropriate graph for the LocalSearch process.
             *
             * Since LocalSearch does not touch the Graph to perform its
             * optimization, this function is provided in case the user wants
             * to cache and operate directly on the Graph itself without having
             * to rebuild it multiple times.
             *
             * @param A The action space of the agents.
             * @param rules An iterable object over QFunctionRules.
             *
             * @return A Graph to perform LocalSearch on.
             */
            template <typename Iterable>
            static Graph makeGraph(const Action & A, const Iterable & inputRules) {
                Graph graph(A.size());

                for (const auto & rule : inputRules) {
                    auto & factorNode = graph.getFactor(rule.action.first)->getData();

                    if (factorNode.size() == 0) {
                        factorNode.resize(factorSpacePartial(rule.action.first, A));
                        factorNode.setZero();
                    }

                    const auto id = toIndexPartial(A, rule.action);
                    factorNode[id] += rule.value;
                }
                return graph;
            }

            /**
             * @brief Basic constructor.
             */
            LocalSearch();

            /**
             * @brief This function approximately finds the best Action-value pair for the provided QFunctionRules.
             *
             * This function automatically sets up the Graph to perform local
             * search on from an iterable of QFunctionRules.
             *
             * This function optimizes over a single randomly sampled initial action.
             *
             * @param A The action space of the agents.
             * @param rules An iterable object over QFunctionRules.
             *
             * @return A tuple containing the best Action and its value over the input rules.
             */
            template <typename Iterable>
            Result operator()(const Action & A, const Iterable & inputRules) {
                Action startAction = makeRandomValue(A, rnd_);

                return (*this)(A, inputRules, startAction);
            }

            /**
             * @brief This function approximately finds the best Action-value pair for the provided QFunctionRules.
             *
             * This function automatically sets up the Graph to perform local
             * search on from an iterable of QFunctionRules.
             *
             * This function optimizes over the input joint action.
             *
             * @param A The action space of the agents.
             * @param rules An iterable object over QFunctionRules.
             * @param startAction The initial action to optimize.
             *
             * @return A tuple containing the best Action and its value over the input rules.
             */
            template <typename Iterable>
            Result operator()(const Action & A, const Iterable & inputRules, Action startAction) {
                Graph graph = makeGraph(A, inputRules);

                return (*this)(A, graph, std::move(startAction));
            }

            /**
             * @brief This function performs the actual local search process.
             *
             * This function optimizes over a single randomly sampled initial action.
             *
             * \sa operator(const Action &, Graph &, Action)
             *
             * @param A The action space of the agents.
             * @param graph The graph to perform local search on.
             *
             * @return The pair for best Action and its value given the internal graph.
             */
            Result operator()(const Action & A, const Graph & graph);

            /**
             * @brief This function performs the actual local search process.
             *
             * We randomly iterate over each agent. Each agent is set to take
             * the action that maximizes the value of the full joint action. We
             * repeat this process until no agent can modify its action to
             * improve the final value.
             *
             * Note that this process is approximate, as it can converge to a
             * local optimum.
             *
             * This function optimizes over the input joint action.
             *
             * @param A The action space of the agents.
             * @param graph The graph to perform local search on.
             * @param startAction The initial action to optimize.
             *
             * @return The pair for best Action and its value given the internal graph.
             */
            Result operator()(const Action & A, const Graph & graph, Action startAction);

        private:
            /**
             * @brief This function evaluates the full score of a given joint action.
             *
             * @param A The action space of the agents.
             * @param graph The graph to compute the score on.
             * @param jointAction The current full joint action.
             *
             * @return The score of the joint action.
             */
            double evaluateGraph(const Action & A, const Graph & graph, const Action & jointAction) const;

            /**
             * @brief This function evaluates the score for a subset of factors in a Graph.
             *
             * Since we only optimize a single agent at a time, there is no
             * reason to re-evaluate the whole graph at each optimization step.
             * This function evaluates only the factors that are needed in the
             * Graph, given the current joint action.
             *
             * @param A The action space of the agents.
             * @param factors The factors to consider for the score.
             * @param jointAction The current full joint action.
             *
             * @return The score of the joint action within the input factors.
             */
            double evaluateFactors(const Action & A, const Graph::FactorItList & factors, const Action & jointAction) const;

            /**
             * @brief This function evaluates the score for a single factor in a Graph.
             *
             * @param A The action space of the agents.
             * @param factor The factor to consider for the score.
             * @param jointAction The current full joint action.
             *
             * @return The score of the joint action within the input factor.
             */
            double evaluateFactor(const Action & A, const Graph::FactorNode & factor, const Action & jointAction) const;

            // Storage for agent ordering (which is shuffled).
            std::vector<size_t> agents_;

            mutable RandomEngine rnd_;
    };
}

#endif
