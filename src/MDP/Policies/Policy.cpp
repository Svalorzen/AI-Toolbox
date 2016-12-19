#include <AIToolbox/MDP/Policies/Policy.hpp>

#include <algorithm>
#include <iostream>

#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        Policy::Policy(const size_t s, const size_t a) :
                PolicyInterface(s, a), policy_(boost::extents[S][A])
        {
            // Random policy is default
            std::fill(policy_.data(), policy_.data() + policy_.num_elements(), 1.0/getA());
        }

        Policy::Policy(const PolicyInterface & p) :
                PolicyInterface(p.getS(), p.getA()), policy_(boost::extents[S][A])
        {
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    policy_[s][a] = p.getActionProbability(s, a);
        }

        Policy::Policy(const size_t s, const size_t a, const ValueFunction & v) :
                PolicyInterface(s, a), policy_(boost::extents[S][A])
        {
            const auto & actions = std::get<ACTIONS>(v);
            for ( size_t s = 0; s < S; ++s )
                policy_[s][actions[s]] = 1.0;
        }

        size_t Policy::sampleAction(const size_t & s) const {
            return sampleProbability(A, policy_[s], rand_);
        }

        double Policy::getActionProbability(const size_t & s, const size_t & a) const {
            return policy_[s][a];
        }

        std::vector<double> Policy::getStatePolicy(const size_t s) const {
            std::vector<double> statePolicy(A);

            std::copy(std::begin(policy_[s]), std::end(policy_[s]), std::begin(statePolicy));

            return statePolicy;
        }

        void Policy::setStatePolicy(const size_t s, const size_t a) {
            for ( size_t ax = 0; ax < A; ++ax )
                policy_[s][ax] = static_cast<double>( ax == a );
        }

        const Policy::PolicyTable & Policy::getPolicyTable() const {
            return policy_;
        }

        void Policy::prettyPrint(std::ostream & os) const {
            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    if ( policy_[s][a] )
                        os << s << "\t" << a << "\t" << std::fixed << policy_[s][a] << "\n";
                }
            }
        }
    }
}
