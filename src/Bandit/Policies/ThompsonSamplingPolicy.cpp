#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <random>

namespace AIToolbox::Bandit {
    ThompsonSamplingPolicy::ThompsonSamplingPolicy(const QFunction & q, const Vector & M2s, const std::vector<unsigned> & counts) :
            Base(q.size()), q_(q), M2s_(M2s), counts_(counts) {}

    size_t ThompsonSamplingPolicy::sampleAction() const {
        // For each arm, we sample its mean. Note that here we use a
        // standardized Student-t distribution, which we then scale using our
        // QFunction and counts parameters to obtain the correct mean
        // estimates.
        size_t bestAction = 0;
        double bestValue;
        {
            // We need at least 2 samples per arm with student-t to estimate
            // the variance.
            if (counts_[0] < 2)
                return 0;

            //     mu = est_mu - t * s / sqrt(n)
            // where
            //     s^2 = 1 / (n-1) * sum_i (x_i - est_mu)^2
            // and
            //     t = student_t sample with n-1 degrees of freedom
            std::student_t_distribution<double> dist(counts_[0] - 1);
            bestValue = q_[0] - dist(rand_) * std::sqrt(M2s_[0] / (counts_[0] * (counts_[0] - 1)));
        }
        for (size_t a = 1; a < A; ++a) {
            if (counts_[a] < 2)
                return a;

            std::student_t_distribution<double> dist(counts_[a] - 1);
            const double val = q_[a] - dist(rand_) * std::sqrt(M2s_[a] / (counts_[a] * (counts_[a] - 1)));

            if (val > bestValue) {
                bestAction = a;
                bestValue = val;
            }
        }

        return bestAction;
    }

    double ThompsonSamplingPolicy::getActionProbability(const size_t & a) const {
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

    Vector ThompsonSamplingPolicy::getPolicy() const {
        // The true formula here is hard, so we don't compute this exactly.
        //
        // Instead we sample, which is easier and possibly faster if we just
        // want a rough approximation.
        constexpr unsigned trials = 100000;

        Vector retval{A};
        retval.setZero();

        for (size_t i = 0; i < trials; ++i)
            retval[sampleAction()] += 1.0;

        retval /= retval.sum();
        return retval;
    }
}
