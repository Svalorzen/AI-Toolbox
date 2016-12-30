#ifndef AI_TOOLBOX_FACTORED_MDP_VARIABLE_ELIMINATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_VARIABLE_ELIMINATION_HEADER_FILE

#include "AIToolbox/ProbabilityUtils.hpp"
#include "AIToolbox/FactoredMDP/Types.hpp"
#include "AIToolbox/FactoredMDP/Utils.hpp"
#include "AIToolbox/FactoredMDP/FactorGraph.hpp"

namespace AIToolbox {
    namespace FactoredMDP {
        /**
         * @brief This class represents the Variable Elimination process.
         *
         * This class performs variable elimination on a factor graph. It first
         * builds the graph starting from a list of QFunctionRules. These rules
         * are sorted by the agents they affect, and each group is added to a
         * single factor connected to those agents.
         *
         * Each agent is then eliminated from the graph, and all rules
         * connected to it are processed in order to find out which action the
         * agent being eliminated should take.
         *
         * When all agents have been eliminated, only the optimal rules
         * containing the best actions remain. The ones that provide the best
         * reward are joined into a single Action, which is then returned.
         *
         * This process is exponential in the maximum number of agents found
         * attached to the same factor (which could be higher than in the
         * original graph, as the elimination process can create bigger factors
         * than the initial ones). However, given that each factor is usually
         * linked to few agents, and that this process allows avoiding
         * considering the full factored Action at any one time, it is usually
         * much faster than a brute-force approach.
         */
        class VariableElimination {
            public:
                // action for subset of agents, value of rule, tags
                using Rule = std::tuple<PartialAction, double, PartialAction>;
                using Rules = std::vector<Rule>;

                struct Factor {
                    Rules rules_;
                };

                using Graph = FactorGraph<Factor>;

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor initializes the internal graph with the
                 * number of needed agents.
                 *
                 * @param a The action space.
                 */
                VariableElimination(Action a);

                /**
                 * @brief This function finds the best Action-value pair for the provided QFunctionRules.
                 *
                 * @param rules An iterable object over QFunctionRules.
                 *
                 * @return A tuple containing the best Action and its value over the input rules.
                 */
                template <typename Iterable>
                std::tuple<Action, double> operator()(const Iterable & rules) {
                    // Should we reset the graph?
                    for (const QFunctionRule & rule : rules) {
                        auto it = graph_.getFactor(rule.a_.first);
                        it->f_.rules_.emplace_back(rule.a_, rule.value_, PartialAction());
                    }
                    return start();
                }

            private:
                /**
                 * @brief This function performs the actual agent elimination process.
                 *
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
                 * @return The pair for best Action and its value given the internal graph.
                 */
                std::pair<Action, double> start();

                /**
                 * @brief This function performs the elimination of a single agent (and all factors next to it) from the internal graph.
                 *
                 * This function adds the resulting best rules which do not
                 * depend on the eliminated action to the remaining factors.
                 *
                 * \sa start()
                 *
                 * @param agent The index of the agent to be removed from the graph.
                 */
                void removeAgent(size_t agent);

                Graph graph_;
                Action A;
                std::vector<Rules> finalFactors_;
        };
    }
}

#endif
