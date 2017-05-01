#include <AIToolbox/FactoredMDP/Algorithms/LLR.hpp>

#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        LLR::LLR(Action a, const std::vector<Factors> & dependencies) :
                A(std::move(a)), L(1), timestep_(0), graph_(A.size()),
                missingExplorations_(*std::max_element(std::begin(A), std::end(A)))
        {
            // Note: L = 1 since we only do 1 action at a time.

            // Build graph keeping all averages.
            for (const auto & dependency : dependencies) {
                auto it = graph_.getFactor(dependency);

                // We already have this one.
                if (it->getData().averages.size() != 0) continue;

                it->getData().averages.resize(factorSpacePartial(dependency, A));
            }
        }

        Action LLR::stepUpdateQ(const Action & a, const Rewards & rew) {
            const auto begin = graph_.factorsBegin();
            const auto end   = graph_.factorsEnd();

            // Update all averages with what we've learned this step.
            // Note: We know that the factors are going to be in the correct
            // order when looping here since we are looping in the same order
            // in which we have inserted them into the graph. So we can
            // correctly match the rewards with the agent groups!
            size_t i = 0;
            for (auto it = begin; it != end; ++it) {
                const auto & agents = graph_.getNeighbors(it);
                // Get previous data
                auto & avg = it->getData().averages[toIndexPartial(agents, A, a)];

                avg.value += (rew[i++] - avg.value) / (++avg.count);
            }
            ++timestep_;
            if (missingExplorations_) {
                Action action(A.size(), 0);
                for (size_t a = 0; a < A.size(); ++a)
                    if (A[a] >= missingExplorations_)
                        action[a] = missingExplorations_ - 1;

                --missingExplorations_;
                return action;
            }

            // Build the vectors to pass to VE
            std::vector<QFunctionRule> rules;
            for (auto it = begin; it != end; ++it) {
                const auto & agents = graph_.getNeighbors(it);

                // std::cout << "Working for "<<agents.size()<< "agents\n";

                PartialFactorsEnumerator enumerator(A, agents);
                while (enumerator.isValid()) {
                    const auto & pAction = *enumerator;
                    // Get the average structure for this action
                    const auto & avg = it->getData().averages[toIndexPartial(A, pAction)];
                    // Create new vector
                    rules.emplace_back(
                        PartialState{}, pAction,
                        avg.value + std::sqrt((L+1) * std::log(timestep_) / avg.count)
                    );
                    enumerator.advance();
                }
            }

            VariableElimination ve(A);
            return std::get<0>(ve(rules));
        }
    }
}
