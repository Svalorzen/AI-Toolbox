#include <AIToolbox/FactoredMDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        QGreedyPolicy::QGreedyPolicy(State s, Action a, const FactoredContainer<QFunctionRule> & q) :
                Base(s, a), q_(q) {}

        Action QGreedyPolicy::sampleAction(const State & s) const {
            VariableElimination ve(A);
            auto rules = q_.filter(s, 0); // Partial filter
            return ve(rules).first;
        }

        double QGreedyPolicy::getActionProbability(const State & s, const Action & a) const {
            if (veccmp(a, sampleAction(s)) == 0) return 1.0;
            return 0.0;
        }
    }
}
