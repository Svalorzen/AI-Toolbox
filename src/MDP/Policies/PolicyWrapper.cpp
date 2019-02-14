#include <AIToolbox/MDP/Policies/PolicyWrapper.hpp>

#include <algorithm>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
    PolicyWrapper::PolicyWrapper(const PolicyMatrix & p) :
            PolicyInterface::Base(p.rows(), p.cols()), policy_(p) {}

    size_t PolicyWrapper::sampleAction(const size_t & s) const {
        return sampleProbability(A, policy_.row(s), rand_);
    }

    double PolicyWrapper::getActionProbability(const size_t & s, const size_t & a) const {
        return policy_(s, a);
    }

    const PolicyWrapper::PolicyMatrix & PolicyWrapper::getPolicyMatrix() const {
        return policy_;
    }

    Matrix2D PolicyWrapper::getPolicy() const {
        return policy_;
    }
}
