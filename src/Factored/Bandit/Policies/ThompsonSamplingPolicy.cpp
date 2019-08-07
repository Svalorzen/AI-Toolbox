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
            const bool isFilled = factorNode.size() > 0;

            if (!isFilled) factorNode.reserve(basis.values.size());

            for (size_t y = 0; y < static_cast<size_t>(basis.values.size()); ++y) {
                double val;
                if (counts[y] < 2) {
                    if (isFilled) continue;
                    // We divide the value by the number of groups_ here with
                    // the hope that the value itself is still high enough that
                    // it shadows the rest of the rules, but it also allows to
                    // sum and compare them so that we still get to optimize
                    // multiple actions at once (the max would just cap to inf).
                    val = std::numeric_limits<double>::max() / q_.bases.size();
                } else {
                    //     mu = est_mu - t * s / sqrt(n)
                    // where
                    //     s^2 = 1 / (n-1) * sum_i (x_i - est_mu)^2
                    // and
                    //     t = student_t sample with n-1 degrees of freedom
                    std::student_t_distribution<double> dist(counts[y] - 1);
                    val = basis.values[y] - dist(rand_) * std::sqrt(m2[y]/(counts[y] * (counts[y] - 1)));
                }

                if (isFilled)
                    factorNode[y].second.first += val;
                else
                    factorNode.emplace_back(y, VE::Factor{val, {}});
            }
        }
        VE ve;
        return std::get<0>(ve(A, graph));
    }

    double ThompsonSamplingPolicy::getActionProbability(const Action & a) const {
        // The true formula here is hard, so we don't compute this exactly.
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
