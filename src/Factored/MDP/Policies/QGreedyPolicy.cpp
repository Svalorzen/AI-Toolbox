#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::MDP {
    QGreedyPolicy::QGreedyPolicy(State s, Action a, const FactoredContainer<QFunctionRule> & q) :
            Base(std::move(s), std::move(a)), qc_(&q), qm_(nullptr) {}

    QGreedyPolicy::QGreedyPolicy(State s, Action a, const FactoredMatrix2D & q) :
            Base(std::move(s), std::move(a)), qc_(nullptr), qm_(&q) {}

    Action QGreedyPolicy::sampleAction(const State & s) const {
        Bandit::VariableElimination ve(A);
        if (qc_) {
            const auto rules = qc_->filter(s, 0); // Partial filter
            return std::get<0>(ve(rules));
        } else {
            std::vector<Bandit::QFunctionRule> rules;
            for (const auto & basis : qm_->bases) {
                PartialFactorsEnumerator se(S, basis.tag);
                PartialFactorsEnumerator ae(A, basis.actionTag);
                for (size_t x = 0; se.isValid(); se.advance(), ++x) {
                    if (!match(s, *se)) continue;

                    ae.reset();
                    for (size_t y = 0; ae.isValid(); ae.advance(), ++y)
                        rules.emplace_back(*ae, basis.values(x, y));
                }
            }
            return std::get<0>(ve(rules));
        }
    }

    double QGreedyPolicy::getActionProbability(const State & s, const Action & a) const {
        if (veccmp(a, sampleAction(s)) == 0) return 1.0;
        return 0.0;
    }
}
