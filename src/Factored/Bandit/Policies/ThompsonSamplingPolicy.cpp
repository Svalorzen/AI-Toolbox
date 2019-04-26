#include <AIToolbox/Factored/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <random>

namespace AIToolbox::Factored::Bandit {
    ThompsonSamplingPolicy::ThompsonSamplingPolicy(const Action & A, const QFunction & q, const std::vector<std::vector<unsigned>> & counts) :
            Base(A), q_(q), counts_(counts) {}

    Action ThompsonSamplingPolicy::sampleAction() const {
        using VE = Bandit::VariableElimination;
        VE::GVE::Graph graph(A.size());

        for (size_t i = 0; i < q_.bases.size(); ++i) {
            const auto & basis = q_.bases[i];
            const auto & counts = counts_[i];
            auto & factorNode = graph.getFactor(basis.tag)->getData();

            factorNode.reserve(basis.values.cols());

            for (size_t y = 0; y < static_cast<size_t>(basis.values.cols()); ++y) {
                std::normal_distribution<double> dist(basis.values(i), 1.0 / (counts[i] + 1));
                factorNode.emplace_back(y, VE::Factor{dist(rand_), {}});
            }
        }
        VE ve;
        return std::get<0>(ve(A, graph));
    }

    double ThompsonSamplingPolicy::getActionProbability(const Action & a) const {
        // The true formula here would be:
        //
        // \int_{-infty, +infty} PDF(N(a)) * CDF(N(0)) * ... * CDF(N(A-1))
        //
        // Where N(x) means the normal distribution obtained from the
        // parameters of that action.
        //
        // Instead we sample, which is easier and possibly faster if we just
        // want a rough approximation.
        constexpr unsigned trials = 1000;
        unsigned selected = 0;

        for (size_t i = 0; i < trials; ++i)
            if (sampleAction() == a)
                ++selected;

        return static_cast<double>(selected) / trials;
    }
}
