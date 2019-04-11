#include <AIToolbox/Bandit/Policies/RandomPolicy.hpp>

namespace AIToolbox::Bandit {
    RandomPolicy::RandomPolicy(const size_t a) :
            PolicyInterface::Base(a),
            randomDistribution_(0, this->A-1) {}

    size_t RandomPolicy::sampleAction() const {
        return randomDistribution_(rand_);
    }

    double RandomPolicy::getActionProbability(const size_t &) const {
        return 1.0/getA();
    }

    Vector RandomPolicy::getPolicy() const {
        Vector p(getA());
        p.fill(1.0/getA());
        return p;
    }
}
