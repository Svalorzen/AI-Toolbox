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
                const auto & pa_v = getBestRule(f);

                a_v.second += pa_v.second;
                for (size_t i = 0; i < pa_v.first.first.size(); ++i)
                    a_v.first[pa_v.first.first[i]] = pa_v.first.second[i];
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
                size_t bestAction = 0;

                for (size_t agentAction = 0; agentAction < A[agent]; ++agentAction) {
                    jointAction.second[id] = agentAction;

                    double newPayoff = 0.0;
                    for (const auto factor : factors)
                        newPayoff += getPayoff(factor->f_, jointAction); // Crossum x 1

                    // We only select the agent best agent action.
                    if (newPayoff > bestPayoff) {
                        bestPayoff = newPayoff;
                        bestAction = agentAction;
                    }
                }
                if (checkDifferentSmall(bestPayoff, 0.0)) {
                    jointAction.second[id] = bestAction;
                    newRules.emplace_back(std::make_pair(jointAction, bestPayoff));
                }
                jointActions.advance();
            }

            for (auto & it : factors)
                graph_.erase(it);
            graph_.erase(agent);

            if (newRules.size() == 0) return;
            if (agents.size() > 1 || agents[0] != agent) {
                agents.erase(std::remove(std::begin(agents), std::end(agents), agent), std::end(agents));

                auto newFactor = graph_.getFactor(agents);
                newFactor->f_.rules_.insert(
                        std::end(newFactor->f_.rules_),
                        std::make_move_iterator(std::begin(newRules)),
                        std::make_move_iterator(std::end(newRules))
                );
            }
            else {
                finalFactors_.push_back(newRules);
            }
        }

        const VariableElimination::Rule & VariableElimination::getBestRule(const Rules & rules) {
            const Rule * bestRule = &rules[0];

            for (const auto & rule : rules)
                if (rule.second > bestRule->second)
                    bestRule = &rule;

            return *bestRule;
        }

        double VariableElimination::getPayoff(const Factor & factor, const PartialAction & jointAction) {
            // TODO: Put a FactoredContaner in each factor?
            double result = 0.0;
            for (const auto & rule : factor.rules_)
                if (match(jointAction, rule.first))
                    result += rule.second;
            return result;
        }
    }
}
