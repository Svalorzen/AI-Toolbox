#include <AIToolbox/MDP/Policies/Policy.hpp>

#include <algorithm>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::MDP {
    Policy::Policy(const size_t s, const size_t a) :
            PolicyInterface::Base(s, a), PolicyWrapper(policy_), policy_(S, A)
    {
        // Random policy is default
        policy_.fill(1.0/getA());
    }

    Policy::Policy(const PolicyInterface::Base & p) :
            PolicyInterface::Base(p.getS(), p.getA()), PolicyWrapper(policy_), policy_(S, A)
    {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                policy_(s, a) = p.getActionProbability(s, a);
    }

    Policy::Policy(const PolicyInterface & p) :
            PolicyInterface::Base(p.getS(), p.getA()), PolicyWrapper(policy_), policy_(p.getPolicy()) {}

    Policy::Policy(const size_t s, const size_t a, const ValueFunction & v) :
            PolicyInterface::Base(s, a), PolicyWrapper(policy_), policy_(S, A)
    {
        const auto & actions = v.actions;
        policy_.setZero();
        for ( size_t s = 0; s < S; ++s )
            policy_(s, actions[s]) = 1.0;
    }

    Policy::Policy(const PolicyMatrix & p) :
            PolicyInterface::Base(p.rows(), p.cols()), PolicyWrapper(policy_), policy_(p)
    {
        for ( size_t s = 0; s < S; ++s )
            if (checkDifferentSmall(policy_.row(s).sum(), 1.0))
                throw std::invalid_argument("Initializing Policy with invalid PolicyMatrix");
    }
}
