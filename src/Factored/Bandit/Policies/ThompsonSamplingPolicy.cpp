#include <AIToolbox/Factored/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>
#include <random>

namespace AIToolbox::Factored::Bandit {
    ThompsonSamplingPolicy::ThompsonSamplingPolicy(const Action & A, const QFunction & q, const std::vector<Vector> & M2s, const std::vector<std::vector<unsigned>> & counts) :
            Base(A), q_(q), M2s_(M2s), counts_(counts) {}

    Action ThompsonSamplingPolicy::sampleAction() const {
        using VE = Bandit::VariableElimination;
        VE::GVE::Graph graph(A.size());

        for (size_t i = 0; i < q_.bases.size(); ++i) {
            const auto & basis = q_.bases[i];
            const auto & m2 = M2s_[i];
            const auto & counts = counts_[i];
            auto & factorNode = graph.getFactor(basis.tag)->getData();

            factorNode.reserve(basis.values.size());

            for (size_t y = 0; y < static_cast<size_t>(basis.values.size()); ++y) {
                if (counts[y] < 2) {
                    factorNode.emplace_back(y, VE::Factor{std::numeric_limits<double>::max(), {}});
                } else {
                    //     mu = est_mu - t * s / sqrt(n)
                    // where
                    //     s^2 = 1 / (n-1) * sum_i (x_i - est_mu)^2
                    // and
                    //     t = student_t sample with n-1 degrees of freedom
                    std::student_t_distribution<double> dist(counts[y] - 1);
                    const auto value = basis.values[y] - dist(rand_) * std::sqrt(m2[y]/(counts[y] * (counts[y] - 1)));

                    factorNode.emplace_back(y, VE::Factor{value, {}});
                }
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
