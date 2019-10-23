#include <AIToolbox/Factored/Bandit/Policies/LLRPolicy.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::Bandit {
    LLRPolicy::LLRPolicy(const Experience & exp) :
            Base(exp.getA()), exp_(exp), L(1)
    {
        // Note: L = 1 since we only do 1 action at a time.
    }

    Action LLRPolicy::sampleAction() const {
        using VE = VariableElimination;

        const auto & A = exp_.getA();

        const auto LtLog = (L+1) * std::log(exp_.getTimesteps());

        VE::GVE::Graph graph(A.size());
        const auto & q = exp_.getRewardMatrix();
        const auto & c = exp_.getVisitsTable();

        for (size_t x = 0; x < q.bases.size(); ++x) {
            const auto & basis = q.bases[x];
            const auto & cc = c[x];
            auto & factorNode = graph.getFactor(basis.tag)->getData();
            const bool isFilled = factorNode.size() > 0;

            if (!isFilled) factorNode.reserve(basis.values.size());

            for (size_t y = 0; y < static_cast<size_t>(basis.values.size()); ++y) {
                // We give rules we haven't seen yet a headstart so they'll get picked first
                // We divide by the number of groups_ here with the hope that the
                // value itself is still high enough that it shadows the rest of
                // the rules, but it also allows to sum and compare them so that we
                // still get to optimize multiple actions at once (the max would
                // just cap to inf).
                double val;
                if (cc[y] == 0) {
                    if (isFilled) continue;
                    val = std::numeric_limits<double>::max() / q.bases.size();
                } else {
                    val = basis.values(y) + std::sqrt(LtLog / cc[y]);
                }

                if (isFilled) {
                    factorNode[y].second.first += val;
                } else {
                    factorNode.emplace_back(y, VE::Factor{val, {}});
                }
            }
        }

        VariableElimination ve;
        return std::get<0>(ve(A, graph));
    }

    double LLRPolicy::getActionProbability(const Action & a) const {
        if (veccmp(a, sampleAction()) == 0) return 1.0;
        return 0.0;
    }

    const Experience & LLRPolicy::getExperience() const {
        return exp_;
    }
}
