#include <AIToolbox/Bandit/Policies/T3CPolicy.hpp>

#include <random>

namespace AIToolbox::Bandit {
    T3CPolicy::T3CPolicy(const Experience & exp, const double beta, const double var) :
            Base(exp.getRewardMatrix().size()), policy_(exp), beta_(beta), var_(var) {}

    size_t T3CPolicy::sampleAction() const {
        const auto & exp = policy_.getExperience();
        const auto & means = exp.getRewardMatrix();
        const auto & counts = exp.getVisitsTable();

        size_t bestAction = policy_.sampleAction();

        if (counts[bestAction] < 2) return bestAction;

        std::bernoulli_distribution pickBest(beta_);
        if (pickBest(rand_))
            return bestAction;

        size_t secondBestAction = 0;
        double lowestCost = std::numeric_limits<double>::max();

        // How many we found with the same lowestCost (to randomly break ties)
        unsigned k;

        for (size_t a = 0; a < getA(); ++a) {
            if (a == bestAction) continue;

            // Compute cost of this action w.r.t. bestAction.
            // - 0.0 if this action has a higher mean.
            // - T3C formula for normals otherwise.
            const double W = (means[a] >= means[bestAction]) ? 0.0 :
                    std::pow(means[bestAction] - means[a], 2) / (2 * var_ * (1.0 / counts[bestAction] + 1.0 / counts[a]));

            if (W < lowestCost) {
                lowestCost = W;
                secondBestAction = a;
                k = 1;
            } else if (W == lowestCost) {
                // Uniformly sample from equal cost alternatives
                std::bernoulli_distribution replace(1.0 / ++k);
                if (replace(rand_)) {
                    lowestCost = W;
                    secondBestAction = a;
                }
            }
        }

        return secondBestAction;
    }

    size_t T3CPolicy::recommendAction() const {
        const auto & exp = policy_.getExperience();

        size_t retval;
        exp.getRewardMatrix().maxCoeff(&retval);

        return retval;
    }

    double T3CPolicy::getActionProbability(const size_t & a) const {
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

    Vector T3CPolicy::getPolicy() const {
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

    const Experience & T3CPolicy::getExperience() const {
        return policy_.getExperience();
    }
}
