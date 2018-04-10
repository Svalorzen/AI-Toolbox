#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::MDP {
    QGreedyPolicy::QGreedyPolicy(State s, Action a, const FactoredContainer<QFunctionRule> & q) :
            Base(std::move(s), std::move(a)), q_(q) {}

    Action QGreedyPolicy::sampleAction(const State & s) const {
        const auto rules = q_.filter(s, 0); // Partial filter
        Bandit::VariableElimination ve(A);
        return std::get<0>(ve(rules));
    }

    double QGreedyPolicy::getActionProbability(const State & s, const Action & a) const {
        if (veccmp(a, sampleAction(s)) == 0) return 1.0;
        return 0.0;
    }
}
