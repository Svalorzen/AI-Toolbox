#include <AIToolbox/Factored/Bandit/Algorithms/LLR.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::Bandit {
    LLR::LLR(Action a, const std::vector<Factors> & dependencies) :
            A(std::move(a)), L(1), timestep_(0), rules_(A)
    {
        // Note: L = 1 since we only do 1 action at a time.

        // Build single rules for each dependency group.
        // This allows us to allocate the rules_ only once, and to just
        // update their values at each timestep.
        for (const auto & agents : dependencies) {
            PartialFactorsEnumerator enumerator(A, agents);
            while (enumerator.isValid()) {
                const auto & pAction = *enumerator;

                rules_.emplace(pAction, pAction, 0.0);

                enumerator.advance();
            }
        }
        averages_.resize(rules_.size());
    }

    Action LLR::stepUpdateQ(const Action & a, const Rewards & rew) {
        // We use the rules_'s Trie in order to obtain the ids of the
        // actions we need to update. We keep in sync the rules_ container
        // with the averages_ vector, so that we can use the Trie's ids for
        // both.
        const auto filtered = rules_.getTrie().filter(a, 0);
        size_t i = 0;
        // We update the averages/counts based on the obtained rewards.
        for (const auto id : filtered)
            averages_[id].value += (rew[i++] - averages_[id].value) / (++averages_[id].count);

        ++timestep_;
        const auto LtLog = (L+1) * std::log(timestep_);

        // Otherwise, recompute all rules' values based on the new timestep
        // and counts.
        for (size_t i = 0; i < rules_.size(); ++i) {
            // We give rules we haven't seen yet a headstart so they'll get picked first
            if (averages_[i].count == 0)
                rules_[i].value = 1000000.0;
            else
                rules_[i].value = averages_[i].value + std::sqrt(LtLog / averages_[i].count);
        }

        VariableElimination ve;
        return std::get<0>(ve(A, rules_));
    }

    FactoredContainer<QFunctionRule> LLR::getQFunctionRules() const {
        auto rulesCopy = rules_;
        for (size_t i = 0; i < rulesCopy.size(); ++i)
            rulesCopy[i].value = averages_[i].value;
        return rulesCopy;
    }
}
