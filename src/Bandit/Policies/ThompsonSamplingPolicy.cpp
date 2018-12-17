#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <random>

namespace AIToolbox::Bandit {
    ThompsonSamplingPolicy::ThompsonSamplingPolicy(const QFunction & q, const std::vector<unsigned> & counts) :
            Base(q.size()), q_(q), counts_(counts) {}

    size_t ThompsonSamplingPolicy::sampleAction() const {
        size_t bestAction = 0;
        double bestValue;
        {
            std::normal_distribution<double> dist(q_[0], 1.0 / (counts_[0] + 1));
            bestValue = dist(rand_);
        }
        for ( size_t a = 1; a < A; ++a ) {
            std::normal_distribution<double> dist(q_[a], 1.0 / (counts_[a] + 1));
            const auto val = dist(rand_);

            if ( val > bestValue ) {
                bestAction = a;
                bestValue = val;
            }
        }

        return bestAction;
    }

    double ThompsonSamplingPolicy::getActionProbability(const size_t & a) const {
        // The true formula here would be:
        //
        // \int_{-infty, +infty} PDF(N(a)) * CDF(N(0)) * ... * CDF(N(A-1))
        //
        // Where N(x) means the normal distribution obtained from the
        // parameters of that action.
        //
        // Instead we sample, which is easier and possibly faster if we just
        // want a rough approximation.
        constexpr unsigned trials = 100000;
        unsigned selected = 0;

        // We avoid recreating the distributions thousands of times here.
        std::vector<std::normal_distribution<double>> dists;
        dists.reserve(A);

        for (size_t aa = 0; aa < A; ++aa)
            dists.emplace_back(q_[aa], 1.0 / (counts_[aa] + 1));

        for (size_t i = 0; i < trials; ++i) {
            size_t bestAction = 0;
            double bestValue = dists[0](rand_);
            for ( size_t aa = 1; aa < A; ++aa ) {
                const auto val = dists[aa](rand_);

                if ( val > bestValue ) {
                    bestAction = aa;
                    bestValue = val;
                }
            }
            if (bestAction == a) ++selected;
        }
        return static_cast<double>(selected) / trials;
    }

    Vector ThompsonSamplingPolicy::getPolicy() const {
        // The true formula here would be:
        //
        // \int_{-infty, +infty} PDF(N(a)) * CDF(N(0)) * ... * CDF(N(A-1))
        //
        // Where N(x) means the normal distribution obtained from the
        // parameters of that action.
        //
        // Instead we sample, which is easier and possibly faster if we just
        // want a rough approximation.
        constexpr unsigned trials = 100000;

        Vector retval{A};
        retval.fill(0.0);

        // We avoid recreating the distributions thousands of times here.
        std::vector<std::normal_distribution<double>> dists;
        dists.reserve(A);

        for (size_t aa = 0; aa < A; ++aa)
            dists.emplace_back(q_[aa], 1.0 / (counts_[aa] + 1));

        for (size_t i = 0; i < trials; ++i) {
            size_t bestAction = 0;
            double bestValue = dists[0](rand_);
            for ( size_t aa = 1; aa < A; ++aa ) {
                const auto val = dists[aa](rand_);

                if ( val > bestValue ) {
                    bestAction = aa;
                    bestValue = val;
                }
            }
            retval[bestAction] += 1.0;
        }
        retval /= retval.sum();
        return retval;
    }
}
