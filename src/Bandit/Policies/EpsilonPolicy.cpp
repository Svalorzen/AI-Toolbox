#include <AIToolbox/Bandit/Policies/EpsilonPolicy.hpp>

namespace AIToolbox::Bandit {
    EpsilonPolicy::EpsilonPolicy(const PolicyInterface & p, double epsilon) :
            PolicyInterface::Base(p.getA()), EpsilonBase(p, epsilon),
            randomDistribution_(0, this->A-1) {}

    size_t EpsilonPolicy::sampleRandomAction() const {
        return randomDistribution_(rand_);
    }

    double EpsilonPolicy::getRandomActionProbability() const {
        return 1.0 / A;
    }

    Vector EpsilonPolicy::getPolicy() const {
        const auto & wrapped = dynamic_cast<const PolicyInterface &>(policy_);
        auto p = wrapped.getPolicy();

        p *= (1.0 - epsilon_);
        p.array() += epsilon_ / A;

        return p;
    }
}
