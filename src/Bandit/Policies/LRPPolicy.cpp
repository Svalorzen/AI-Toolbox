#include <AIToolbox/Bandit/Policies/LRPPolicy.hpp>

#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Bandit {
    LRPPolicy::LRPPolicy(size_t A, double a, double b) :
        Base(A),
        a_(a), invB_(1.0 - b), divB_(b / (A - 1)), policy_(A)
    {
        policy_.fill(1.0 / A);
    }

    void LRPPolicy::stepUpdateP(size_t act, bool result) {
        if (result) {
            policy_[act] += a_ * (1.0 - policy_[act]);
            for (size_t i = 0; i < static_cast<size_t>(policy_.size()); ++i)
                if (i != act)
                    policy_[i] -= a_ * policy_[i];
        } else {
            policy_[act] *= invB_;
            for (size_t i = 0; i < static_cast<size_t>(policy_.size()); ++i)
                if (i != act)
                    policy_[i] = divB_ + invB_ * policy_[i];
        }
    }

    size_t LRPPolicy::sampleAction() const {
        return sampleProbability(A, policy_, rand_);
    }

    double LRPPolicy::getActionProbability(const size_t & a) const {
        return policy_[a];
    }

    Vector LRPPolicy::getPolicy() const {
        return policy_;
    }

    void LRPPolicy::setAParam(double a) { a_ = a; }
    double LRPPolicy::getAParam() const { return a_; }
    void LRPPolicy::setBParam(double b) { invB_ = 1.0 - b; divB_ = b / (A - 1); }
    double LRPPolicy::getBParam() const { return 1.0 - invB_; }
}
