#ifndef AI_TOOLBOX_FACTORED_BANDIT_MULTI_OBJECTIVE_VARIABLE_ELIMINATION_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_MULTI_OBJECTIVE_VARIABLE_ELIMINATION_HEADER_FILE

#include "AIToolbox/Factored/Bandit/Types.hpp"
#include "AIToolbox/Factored/Utils/Core.hpp"
#include "AIToolbox/Factored/Utils/FactorGraph.hpp"

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents the Multi Objective Variable Elimination process.
     *
     * This class performs variable elimination on a factor graph. It first
     * builds the graph starting from a list of MOQFunctionRules. These
     * rules are sorted by the agents they affect, and each group is added
     * to a single factor connected to those agents.
     *
     * Each agent is then eliminated from the graph, and all rules
     * connected to it are processed in order to find out which actions the
     * agent being eliminated should take.
     *
     * When doing multi-objective elimination, there is no real best action
     * in general, since we do not know in advance the weights for the
     * objectives' rewards. Thus, we keep all action-rewards pairs we found
     * during the elimination and return them.
     *
     * This process is exponential in the maximum number of agents found
     * attached to the same factor (which could be higher than in the
     * original graph, as the elimination process can create bigger factors
     * than the initial ones). However, given that each factor is usually
     * linked to few agents, and that this process allows avoiding
     * considering the full factored Action at any one time, it is usually
     * much faster than a brute-force approach.
     *
     * WARNING: This process only considers rules that have been explicitly
     * passed to it. This may create problems if some of your values have
     * negative values in it, since the elimination process will not
     * consider unmentioned actions as giving 0 reward, and choose them
     * instead of the negative values. In order to avoid this problem
     * either all 0 rules have to be explicitly mentioned for each agent
     * subgroup containing negative rules, or the rules have to be
     * converted to an equivalent graph with positive values.
     */
    class MultiObjectiveVariableElimination {
        public:
            using Entry = std::tuple<PartialAction, Rewards>;
            using Entries = std::vector<Entry>;
            using Rule = std::tuple<PartialAction, Entries>;
            using Rules = std::vector<Rule>;

            using Results = Entries;

            struct Factor {
                Rules rules;
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
            MultiObjectiveVariableElimination(Action a);

            /**
             * @brief This function finds the best Action-value pair for the provided MOQFunctionRules.
             *
             * @param rules An iterable object over MOQFunctionRules.
             *
             * @return The pair of best Action and its value over the input rules.
             */
            template <typename Iterable>
            Results operator()(const Iterable & inputRules) {
                // Should we reset the graph?
                for (const auto & rule : inputRules) {
                    auto & rules = graph_.getFactor(rule.action.first)->getData().rules;
                    rules.emplace_back(rule.action, Entries{std::make_tuple(PartialAction(), rule.values)});
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
             * All the responses found (possibly pruned) are added as Rules
             * to a (possibly new) factor adjacent to the adjacent agents.
             *
             * The process is repeated until all agents are eliminated.
             *
             * What remains is then returned.
             *
             * @return All pairs of PartialAction, Rewards found during the elimination process.
             */
            Results start();

            /**
             * @brief This function performs the elimination of a single agent (and all factors next to it) from the internal graph.
             *
             * This function adds the resulting rules which do not depend
             * on the eliminated action to the remaining factors.
             *
             * \sa start()
             *
             * @param agent The index of the agent to be removed from the graph.
             */
            void removeAgent(size_t agent);

            Action A;
            Graph graph_;
            std::vector<Entries> finalFactors_;
    };
}

#endif
