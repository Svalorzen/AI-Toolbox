#include <AIToolbox/EpsilonPolicy.hpp>

#include <stdexcept>

namespace AIToolbox {
    EpsilonPolicy::EpsilonPolicy(const PolicyInterface & p, double e) : PolicyInterface(p.getS(), p.getA()), policy_(p), epsilon_(e), randomDistribution_(0, A-1)
    {
        if ( epsilon_ < 0.0 || epsilon_ > 1.0 ) throw std::runtime_error("Epsilon must be >= 0 and <= 1");
    }

    size_t EpsilonPolicy::sampleAction(size_t s) const {
        double pe = sampleDistribution_(rand_);
        if ( pe > epsilon_ ) {
            return randomDistribution_(rand_);
        }
        return policy_.sampleAction(s);
    }

    double EpsilonPolicy::getActionProbability(size_t s, size_t a) const {
        //          Probability of taking old decision          Other probability
        return epsilon_ * policy_.getActionProbability(s,a) + ( 1.0 - epsilon_ ) / A;
    }

    void EpsilonPolicy::setEpsilon(double e) {
        if ( e < 0.0 || e > 1.0 ) return;
        epsilon_ = e;
    }

    double EpsilonPolicy::getEpsilon() const {
        return epsilon_;
    }

}
