#include <AIToolbox/Factored/Bandit/Policies/QGreedyPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::Bandit {
    QGreedyPolicy::QGreedyPolicy(Action a, const FilterMap<QFunctionRule> & q) :
            Base(std::move(a)), qc_(&q), qm_(nullptr) {}

    QGreedyPolicy::QGreedyPolicy(Action a, const QFunction & q) :
            Base(std::move(a)), qc_(nullptr), qm_(&q) {}

    Action QGreedyPolicy::sampleAction() const {
        using VE = Bandit::VariableElimination;
        VE ve;
        if (qc_) {
            return std::get<0>(ve(A, *qc_));
        } else {
            VE::GVE::Graph graph(A.size());

            for (size_t x = 0; x < qm_->bases.size(); ++x) {
                const auto & basis = qm_->bases[x];
                auto & factorNode = graph.getFactor(basis.tag)->getData();
                const bool isFilled = factorNode.size() > 0;

                if (!isFilled) factorNode.reserve(basis.values.size());

                for (size_t y = 0; y < static_cast<size_t>(basis.values.size()); ++y) {
                    if (isFilled) {
                        factorNode[y].second.first += basis.values(y);
                    } else {
                        factorNode.emplace_back(y, VE::Factor{basis.values(y), {}});
                    }
                }
            }
            return std::get<0>(ve(A, graph));
        }
    }

    double QGreedyPolicy::getActionProbability(const Action & a) const {
        if (veccmp(a, sampleAction()) == 0) return 1.0;
        return 0.0;
    }
}
