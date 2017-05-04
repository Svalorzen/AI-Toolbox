#include <AIToolbox/FactoredMDP/Algorithms/LLR.hpp>

#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        LLR::LLR(Action a, const std::vector<Factors> & dependencies) :
                A(std::move(a)), L(1), timestep_(0), rules_(A),
                missingExplorations_(*std::max_element(std::begin(A), std::end(A)))
        {
            // Note: L = 1 since we only do 1 action at a time.

            // TODO: Fix comments in FactoredContainer
            for (const auto & agents : dependencies) {
                PartialFactorsEnumerator enumerator(A, agents);
                while (enumerator.isValid()) {
                    const auto & pAction = *enumerator;

                    rules_.emplace(pAction, PartialState{}, pAction, 0.0);

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
            auto filtered = rules_.getTrie().filter(a, 0);
            size_t i = 0;
            // We update the averages/counts based on the obtained rewards.
            for (auto id : filtered)
               averages_[id].value += (rew[i++] - averages_[id].value) / (++averages_[id].count);

            ++timestep_;
            // If we haven't yet explored all possible actions, we explore some more.
            // This is so we don't have to divide by 0 later.
            if (missingExplorations_) {
                Action action(A.size(), 0);
                for (size_t a = 0; a < A.size(); ++a)
                    if (A[a] >= missingExplorations_)
                        action[a] = missingExplorations_ - 1;

                --missingExplorations_;
                return action;
            }

            // Otherwise, recompute all rules' values based on the new timestep
            // and counts.
            for (size_t i = 0; i < rules_.size(); ++i)
                rules_[i].value_ = averages_[i].count + std::sqrt((L+1) * std::log(timestep_) / averages_[i].count);

            VariableElimination ve(A);
            return std::get<0>(ve(rules_));
        }

        FactoredContainer<QFunctionRule> LLR::getQFunctionRules() const {
            auto rulesCopy = rules_;
            for (size_t i = 0; i < rulesCopy.size(); ++i)
                rulesCopy[i].value_ = averages_[i].value;
            return rulesCopy;
        }
    }
}
