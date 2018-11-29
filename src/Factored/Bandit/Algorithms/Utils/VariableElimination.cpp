#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    using VE = VariableElimination;

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

    VE::VariableElimination(Action a) : A(std::move(a)), graph_(A.size()) {}

    VE::Result VE::start() {
        // This can possibly be improved with some heuristic ordering
        while (graph_.variableSize())
            removeAgent(graph_.variableSize() - 1);

        auto a_v = std::make_pair(Action(A.size()), 0.0);
        for (const auto & f : finalFactors_) {
            a_v.second += f.first;
            // Add tags together
            const auto & tags = f.second;
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

        const bool isFinalFactor = agents.size() == 1;

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
                    newPayoff += getPayoff(factor->getData().rules, jointAction, &newTag);

                // We only select the agent's best action.
                if (newPayoff > bestPayoff) {
                    bestPayoff = newPayoff;
                    bestTag = std::move(newTag);
                }
            }
            if (checkDifferentGeneral(bestPayoff, std::numeric_limits<double>::lowest())) {
                if (!isFinalFactor) {
                    newRules.emplace_back(removeFactor(jointAction, agent), Entry{bestPayoff, std::move(bestTag)});
                } else {
                    finalFactors_.emplace_back(bestPayoff, std::move(bestTag));
                }
            }
            jointActions.advance();
        }

        for (const auto & it : factors)
            graph_.erase(it);
        graph_.erase(agent);

        if (newRules.size() == 0) return;
        if (!isFinalFactor) {
            agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

            auto newFactor = graph_.getFactor(agents);
            newFactor->getData().rules.insert(
                    std::end(newFactor->getData().rules),
                    std::make_move_iterator(std::begin(newRules)),
                    std::make_move_iterator(std::end(newRules))
            );
        }
    }

    double getPayoff(const VE::Rules & rules, const PartialAction & jointAction, PartialAction * tags) {
        double result = 0.0;
        // Note here that we must use match since the factors adjacent to
        // one agent aren't all next to all its neighbors. Since they are
        // different, we must coarsely check that equal agents do equal
        // actions.
        for (const auto & rule : rules) {
            if (match(jointAction, std::get<0>(rule))) {
                result += std::get<1>(rule).first;
                unsafe_join(tags, std::get<1>(rule).second);
            }
        }
        return result;
    }
}
