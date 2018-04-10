#include <AIToolbox/Factored/Bandit/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::Bandit {
    QGreedyPolicy::QGreedyPolicy(Action a, const FactoredContainer<QFunctionRule> & q) :
            Base(std::move(a)), q_(q) {}

    Action QGreedyPolicy::sampleAction() const {
        Bandit::VariableElimination ve(A);
        return std::get<0>(ve(q_));
    }

    double QGreedyPolicy::getActionProbability(const Action & a) const {
        if (veccmp(a, sampleAction()) == 0) return 1.0;
        return 0.0;
    }
}
