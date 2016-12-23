#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>
#include <boost/functional/hash.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        VariableElimination::VariableElimination(Action a) : graph_(a.size()), A(a) {}

        std::pair<Action, double> VariableElimination::start() {
            // This can possibly be improved with some heuristic ordering
            while (graph_.agentSize())
                removeAgent(graph_.agentSize() - 1);

            auto a_v = std::make_pair(Action(A.size()), 0.0);
            for (const auto & f : finalFactors_) {
                const auto & pa_v_t = getBestRule(f);

                a_v.second += std::get<1>(pa_v_t);
                // Add tags together
                const auto & tags = std::get<2>(pa_v_t);
                for (size_t i = 0; i < tags.first.size(); ++i)
                    a_v.first[tags.first[i]] = tags.second[i];
            }

            return a_v;
        }

        void VariableElimination::removeAgent(size_t agent) {
            auto factors = graph_.getNeighbors(agent);
            auto agents = graph_.getNeighbors(factors);

            Rules newRules;
            PartialFactorsEnumerator jointActions(A, agents, agent);
            auto id = jointActions.getFactorToSkipId();

            while (jointActions.isValid()) {
                auto & jointAction = *jointActions;
                double bestPayoff = 0.0;
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
                        newPayoff += getPayoff(factor->f_, jointAction, &newTag);

                    // We only select the agent's best action.
                    if (newPayoff > bestPayoff) {
                        bestPayoff = newPayoff;
                        bestTag = std::move(newTag);
                    }
                }
                if (checkDifferentSmall(bestPayoff, 0.0)) {
                    newRules.emplace_back(jointAction, bestPayoff, std::move(bestTag));
                }
                jointActions.advance();
            }

            for (auto & it : factors)
                graph_.erase(it);
            graph_.erase(agent);

            if (newRules.size() == 0) return;
            if (agents.size() > 1) {
                agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

                auto newFactor = graph_.getFactor(agents);
                newFactor->f_.rules_.insert(
                        std::end(newFactor->f_.rules_),
                        std::make_move_iterator(std::begin(newRules)),
                        std::make_move_iterator(std::end(newRules))
                );
            } else {
                finalFactors_.push_back(newRules);
            }
        }

        const VariableElimination::Rule & VariableElimination::getBestRule(const Rules & rules) {
            const Rule * bestRule = &rules[0];

            for (const auto & rule : rules)
                if (std::get<1>(rule) > std::get<1>(*bestRule))
                    bestRule = &rule;

            return *bestRule;
        }

        double VariableElimination::getPayoff(const Factor & factor, const PartialAction & jointAction, PartialAction * tags) {
            double result = 0.0;
            for (const auto & rule : factor.rules_) {
                if (match(jointAction, std::get<0>(rule))) {
                    result += std::get<1>(rule);
                    inplace_merge(tags, std::get<2>(rule));
                }
            }
            return result;
        }
    }
}
