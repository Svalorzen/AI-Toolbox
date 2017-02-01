#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        using VE = VariableElimination;

        /**
         * @brief This function finds the highest valued rule in the given rules.
         *
         * @param rules A vector of Rule with at least 1 element.
         *
         * @return The highest valued rule.
         */
        const VE::Rule & getBestRule(const VE::Rules & rules);

        /**
         * @brief This function returns the sum of values of all rules matching the input action.
         *
         * @param rules The rules to be searched in.
         * @param jointAction The joint action to match each Rule against.
         * @param tags An optional pointer where to store all tags encountered in the sum.
         *
         * @return The sum of all matching Rules' values.
         */
        double getPayoff(const VE::Rules & rules, const PartialAction & jointAction, PartialAction * tags = nullptr);

        VE::VariableElimination(Action a) : graph_(a.size()), A(a) {}

        VE::Result VE::start() {
            // This can possibly be improved with some heuristic ordering
            while (graph_.agentSize())
                removeAgent(graph_.agentSize() - 1);

            auto a_v = std::make_pair(Action(A.size()), 0.0);
            for (const auto & f : finalFactors_) {
                const auto & pa_t_v = getBestRule(f);

                a_v.second += std::get<2>(pa_t_v);
                // Add tags together
                const auto & tags = std::get<1>(pa_t_v);
                for (size_t i = 0; i < tags.first.size(); ++i)
                    a_v.first[tags.first[i]] = tags.second[i];
            }

            return a_v;
        }

        void VariableElimination::removeAgent(const size_t agent) {
            const auto factors = graph_.getNeighbors(agent);
            auto agents = graph_.getNeighbors(factors);

            Rules newRules;
            PartialFactorsEnumerator jointActions(A, agents, agent);
            auto id = jointActions.getFactorToSkipId();

            while (jointActions.isValid()) {
                auto & jointAction = *jointActions;
                double bestPayoff = std::numeric_limits<double>::lowest();
                PartialAction bestTag;

                // So here we're trying to create a single rule with a value
                // optimal for this particular joint action for this subset of
                // agents, aside from the one we are going to eliminate.
                //
                // So we're going to try all actions of the agent to be
                // eliminated, and see which one gives us the best return.
                // Once we know, we pick that as the best rule, we add it, and
                // we try the next joint action.
                for (size_t agentAction = 0; agentAction < A[agent]; ++agentAction) {
                    jointAction.second[id] = agentAction;

                    double newPayoff = 0.0;
                    PartialAction newTag{{agent}, {agentAction}};
                    // The idea here is that we sum all values for all factors
                    // touching these agents. In doing so, we also track all
                    // actions of all other agents that contributed in the
                    // creation of those rules. Since those agents are
                    // necessarily all different (since if they weren't they
                    // would have resolved together to a single rule), we can
                    // create a tag with their action by simply writing in it.
                    for (const auto factor : factors)
                        newPayoff += getPayoff(factor->getData().rules_, jointAction, &newTag);

                    // We only select the agent's best action.
                    if (newPayoff > bestPayoff) {
                        bestPayoff = newPayoff;
                        bestTag = std::move(newTag);
                    }
                }
                if (checkDifferentGeneral(bestPayoff, std::numeric_limits<double>::lowest()))
                    newRules.emplace_back(removeFactor(jointAction, agent), std::move(bestTag), bestPayoff);
                jointActions.advance();
            }

            for (const auto & it : factors)
                graph_.erase(it);
            graph_.erase(agent);

            if (newRules.size() == 0) return;
            if (agents.size() > 1) {
                agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

                auto newFactor = graph_.getFactor(agents);
                newFactor->getData().rules_.insert(
                        std::end(newFactor->getData().rules_),
                        std::make_move_iterator(std::begin(newRules)),
                        std::make_move_iterator(std::end(newRules))
                );
            } else {
                finalFactors_.push_back(newRules);
            }
        }

        const VE::Rule & getBestRule(const VE::Rules & rules) {
            const VE::Rule * bestRule = &rules[0];

            for (const auto & rule : rules)
                if (std::get<1>(rule) > std::get<1>(*bestRule))
                    bestRule = &rule;

            return *bestRule;
        }

        double getPayoff(const VE::Rules & rules, const PartialAction & jointAction, PartialAction * tags) {
            double result = 0.0;
            for (const auto & rule : rules) {
                if (match(jointAction, std::get<0>(rule))) {
                    result += std::get<2>(rule);
                    inplace_merge(tags, std::get<1>(rule));
                }
            }
            return result;
        }
    }
}
