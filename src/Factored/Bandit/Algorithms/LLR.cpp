#include <AIToolbox/Factored/Bandit/Algorithms/LLR.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::Bandit {
    LLR::LLR(Action a, const std::vector<PartialKeys> & dependencies) :
            A(std::move(a)), L(1), timestep_(0), averages_(A, dependencies)
    {
        // Note: L = 1 since we only do 1 action at a time.
    }

    Action LLR::stepUpdateQ(const Action & a, const Rewards & rew) {
        using VE = VariableElimination;

        averages_.stepUpdateQ(a, rew);

        ++timestep_;
        const auto LtLog = (L+1) * std::log(timestep_);

        VE::GVE::Graph graph(A.size());
        const auto & q = averages_.getQFunction();
        const auto & c = averages_.getCounts();

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

    const RollingAverage & LLR::getRollingAverage() const {
        return averages_;
    }
}
