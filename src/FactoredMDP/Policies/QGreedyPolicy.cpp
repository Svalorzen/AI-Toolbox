#include <AIToolbox/FactoredMDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        QGreedyPolicy::QGreedyPolicy(State s, Action a, const FactoredContainer<QFunctionRule> & q) :
                Base(std::move(s), std::move(a)), q_(q) {}

        Action QGreedyPolicy::sampleAction(const State & s) const {
            VariableElimination ve(A);
            auto rules = q_.filter(s, 0); // Partial filter
            return std::get<0>(ve(rules));
        }

        double QGreedyPolicy::getActionProbability(const State & s, const Action & a) const {
            if (veccmp(a, sampleAction(s)) == 0) return 1.0;
            return 0.0;
        }
    }
}
