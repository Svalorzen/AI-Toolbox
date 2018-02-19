#include <AIToolbox/MDP/Policies/RandomPolicy.hpp>

namespace AIToolbox::MDP {
    RandomPolicy::RandomPolicy(const size_t s, const size_t a) :
            PolicyInterface::Base(s, a),
            randomDistribution_(0, this->A-1) {}

    size_t RandomPolicy::sampleAction(const size_t &) const {
        return randomDistribution_(rand_);
    }

    double RandomPolicy::getActionProbability(const size_t &, const size_t &) const {
        return 1.0/getA();
    }

    Matrix2D RandomPolicy::getPolicy() const {
        Matrix2D p;
        p.fill(1.0/getA());
        return p;
    }
}
