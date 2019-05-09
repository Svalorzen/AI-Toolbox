#include <AIToolbox/Factored/Bandit/Algorithms/RollingAverage.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    RollingAverage::RollingAverage(Action a, const std::vector<std::vector<size_t>> & dependencies) :
            A(std::move(a))
    {
        qfun_.bases.resize(dependencies.size());
        counts_.resize(dependencies.size());

        for (size_t i = 0; i < qfun_.bases.size(); ++i) {
            qfun_.bases[i].tag = dependencies[i];
            qfun_.bases[i].values.resize(factorSpacePartial(dependencies[i], A));
            qfun_.bases[i].values.setZero();

            counts_[i].resize(qfun_.bases[i].values.size());
        }
    }

    void RollingAverage::stepUpdateQ(const Action & a, const Rewards & rews) {
        assert(rews.size() == qfun_.bases.size());
        for (size_t i = 0; i < qfun_.bases.size(); ++i) {
            const auto aId = toIndexPartial(qfun_.bases[i].tag, A, a);
            auto & v = qfun_.bases[i].values;
            auto c = counts_[i][aId];

            v[aId] = (c * v[aId] + rews[i]) / (c + 1);
            ++counts_[i][aId];
        }
    }

    void RollingAverage::reset() {
        for (auto & basis : qfun_.bases)
            basis.values.setZero();
        for (auto & c : counts_)
            std::fill(std::begin(c), std::end(c), 0);
    }

    const Action & RollingAverage::getA() const { return A; }
    const QFunction & RollingAverage::getQFunction() const { return qfun_; }
    const std::vector<std::vector<unsigned>> & RollingAverage::getCounts() const { return counts_; }
}
