#include <AIToolbox/Factored/Bandit/Policies/RandomPolicy.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    RandomPolicy::RandomPolicy(Action a) :
            Base(std::move(a)), action_(getA().size())
    {
        for (size_t a = 0; a < getA().size(); ++a)
            randomDistributions_.emplace_back(0, getA()[a]-1);
    }

    Action RandomPolicy::sampleAction() const {
        return sampleActionNoAlloc();
    }

    const Action & RandomPolicy::sampleActionNoAlloc() const {
        for (size_t a = 0; a < getA().size(); ++a)
            action_[a] = randomDistributions_[a](rand_);
        return action_;
    }

    double RandomPolicy::getActionProbability(const Action &) const {
        return 1.0/factorSpace(getA());
    }
}
