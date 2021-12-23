#ifndef AI_TOOLBOX_FACTORED_BANDIT_LOCAL_SEARCH_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_LOCAL_SEARCH_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>

namespace AIToolbox::Factored::Bandit {
    class LocalSearch {
        public:
            using Result = std::tuple<Action, double>;
            using Graph = FactorGraph<Vector>;

            LocalSearch();

            /**
             * @brief This function approximately finds the best Action-value pair for the provided QFunctionRules.
             *
             * This function automatically sets up the Graph to perform local
             * search on from an iterable of QFunctionRules.
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

            template <typename Iterable>
            Result operator()(const Action & A, const Iterable & inputRules, Action startAction) {
                Graph graph(A.size());

                for (const auto & rule : inputRules) {
                    auto & factorNode = graph.getFactor(rule.action.first)->getData();

                    if (factorNode.size() == 0) {
                        factorNode.resize(factorSpacePartial(A, rule.action.first));
                        factorNode.setZero();
                    }

                    const auto id = toIndexPartial(A, rule.action);
                    factorNode[id] += rule.value;
                }

                return (*this)(A, graph, std::move(startAction));
            }

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
             * @param A The action space of the agents.
             * @param graph The graph to perform local search on.
             *
             * @return The pair for best Action and its value given the internal graph.
             */
            Result operator()(const Action & A, Graph & graph);
            Result operator()(const Action & A, Graph & graph, Action startAction);

        private:
            double evaluateFactors(const Action & A, const Graph::FactorItList & factors, const Action & jointAction) const;

            std::vector<size_t> agents_;

            mutable RandomEngine rnd_;
    };
}

#endif
