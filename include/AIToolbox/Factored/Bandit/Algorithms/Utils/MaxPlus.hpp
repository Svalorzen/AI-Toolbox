#ifndef AI_TOOLBOX_FACTORED_BANDIT_MAX_PLUS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_MAX_PLUS_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/FactorGraph.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents the Max-Plus optimization algorithm for loopy FactorGraphs.
     *
     * Max-Plus is the equivalent algorithm to the max-product algorithm for
     * Bayesian networks. It is used to (in graphs, approximately) compute the
     * optimal joint action for multiple agents very quickly.
     *
     * Max-Plus works by sending messages between the agents and factors in the
     * FactorGraph representation of the coordination problem. While algorithms
     * like VariableElimination postpone the actual maximization until the end,
     * MaxPlus performs local maximizations repeatedly until convergence. Since
     * local maximizations are performed on relatively small functions (and can
     * possibly be done in parallel), MaxPlus is quite fast, although it cannot
     * guarantee convergence in loopy graphs.
     *
     * Agent nodes simply send to each adjacent factor the sum of all messages
     * received from the other ones (so it excludes the message received by
     * that same factor).
     *
     * Factor nodes add their own original function to the cross-product of all
     * received messages. Then, to each adjacent agent they send a message
     * where all other agents are maximized.
     *
     * The optimal action is selected locally by the agent nodes, by simply
     * selecting the action that maximizes the sum of all received messages.
     * Since in loopy graphs this is not guaranteed to convergence, we only
     * update the returned action if the new overall value is greater than what
     * was selected before.
     *
     */
    class MaxPlus {
        public:
            using Result = std::tuple<Action, double>;

            // Values of factor (in theory N dimensional matrix)
            using Factor = Vector;
            using Graph = FactorGraph<Factor>;

            /**
             * @brief This function finds the best Action-value pair for the provided QFunctionRules.
             *
             * This function automatically sets up the Graph to perform MaxPlus
             * on from an iterable of QFunctionRules.
             *
             * Warning: The returned value is most likely incorrect, and we
             * only return it for informative purpouses. This is because
             * MaxPlus needs to internally normalize the values to avoid
             * divergence, which makes the final computed value incorrect.
             *
             * @param A The action space of the agents.
             * @param rules An iterable object over QFunctionRules.
             * @param iterations The number of message-passing iterations to perform.
             *
             * @return A tuple containing the best Action and its (approximate) value over the input rules.
             */
            template <typename Iterable>
            Result operator()(const Action & A, const Iterable & inputRules, size_t iterations = 10);

            /**
             * @brief This function performs the actual MaxPlus algorithm.
             *
             * We maintain two matrices, inMessages and outMessages, which
             * represent the messages sent by agents, and by factors
             * respectively. At the end of each iteration the two are swapped.
             *
             * At e
             * For each agent, its adjacent factors, and the agents
             * adjacent to those are found. Then all possible action
             * combinations between those other agents are tried in order
             * to find the best action response be for the agent to be
             * eliminated.
             *
             * All the best responses found are added as Rules to a
             * (possibly new) factor adjacent to the adjacent agents.
             *
             * The process is repeated until all agents are eliminated.
             *
             * What remains is then joined into a single Action, containing
             * the best possible value.
             *
             * @param A The action space of the agents.
             * @param graph The graph to perform VE on.
             * @param iterations The number of message-passing iterations to perform.
             *
             * @return The pair for best Action and its value given the internal graph.
             */
            Result operator()(const Action & A, const Graph & graph, size_t iterations = 10);
    };

    template <typename Iterable>
    MaxPlus::Result MaxPlus::operator()(const Action & A, const Iterable & inputRules, size_t iterations) {
        Graph graph(A.size());

        for (const auto & rule : inputRules) {
            auto & factorData = graph.getFactor(rule.action.first)->getData();
            const auto id = toIndexPartial(A, rule.action);

            if (factorData.size() == 0) {
                factorData.resize(factorSpacePartial(rule.action.first, A));
                factorData.setZero();
            }

            factorData[id] = rule.value;
        }

        return (*this)(A, graph, iterations);
    }
}

#endif
