#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

#include <random>

namespace AIToolbox::Bandit {
    ThompsonSamplingPolicy::ThompsonSamplingPolicy(const size_t A) : Base(A), experience_(A) {}

    void ThompsonSamplingPolicy::stepUpdateP(const size_t a, const double reward) {
        auto & [avg, tries] = experience_[a];
        // Rolling average for this bandit arm
        avg = (tries * avg + reward) / (tries + 1);
        ++tries;
    }

    size_t ThompsonSamplingPolicy::sampleAction() const {
        size_t bestAction = 0;
        double bestValue;
        {
            std::normal_distribution<double> dist(experience_[0].first, 1.0 / (experience_[0].second + 1));
            bestValue = dist(rand_);
        }
        for ( size_t a = 1; a < A; ++a ) {
            std::normal_distribution<double> dist(experience_[a].first, 1.0 / (experience_[a].second + 1));
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

        for (size_t i = 0; i < A; ++i)
            dists.emplace_back(experience_[a].first, 1.0 / (experience_[a].second + 1));

        for (size_t i = 0; i < trials; ++i) {
            size_t bestAction = 0;
            double bestValue = dists[0](rand_);
            for ( size_t a = 1; a < A; ++a ) {
                const auto val = dists[a](rand_);

                if ( val > bestValue ) {
                    bestAction = a;
                    bestValue = val;
                }
            }
            if (bestAction == a) ++selected;
        }
        return static_cast<double>(selected) / trials;
    }
}
