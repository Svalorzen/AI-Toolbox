#include <AIToolbox/MDP/Policies/Policy.hpp>

#include <algorithm>
#include <iostream>

#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    Policy::Policy(size_t s, size_t a) : PolicyInterface<size_t>(s, a), policy_(boost::extents[S][A])
    {
        // Random policy is default
        std::fill(policy_.data(), policy_.data() + policy_.num_elements(), 1.0/getA());
    }

    Policy::Policy(const PolicyInterface & p) : PolicyInterface(p.getS(), p.getA()), policy_(boost::extents[S][A])
    {
        for ( size_t s = 0; s < S; ++s )
            for ( size_t a = 0; a < A; ++a )
                policy_[s][a] = p.getActionProbability(s, a);
    }

    size_t Policy::sampleAction(const size_t & s) const {
        return sampleProbability(policy_[s], A, rand_);
    }

    double Policy::getActionProbability(const size_t & s, size_t a) const {
        return policy_[s][a];
    }

    std::vector<double> Policy::getStatePolicy( size_t s ) const {
        std::vector<double> statePolicy(A);

        std::copy(std::begin(policy_[s]), std::end(policy_[s]), std::begin(statePolicy));

        return statePolicy;
    }

    void Policy::setStatePolicy(size_t s, size_t a) {
        for ( size_t ax = 0; ax < A; ++ax )
            policy_[s][ax] = static_cast<double>( ax == a );
    }

    const Policy::PolicyTable & Policy::getPolicy() const {
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

    std::istream& operator>>(std::istream &is, Policy &p) {
        size_t S = p.getS();
        size_t A = p.getA();

        Policy policy(S, A);

        size_t scheck, acheck;

        bool fail = false;
        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t a = 0; a < A; ++a ) {
                if ( ! ( is >> scheck >> acheck >> policy.policy_[s][a] ) ) {
                    std::cerr << "AIToolbox: Could not read policy data.\n"; 
                    fail = true;
                }
                else if ( policy.policy_[s][a] < 0.0 || policy.policy_[s][a] > 1.0 ) {
                    std::cerr << "AIToolbox: Input policy data contains non-probability values.\n";
                    fail = true;
                }
                else if ( scheck != s || acheck != a ) {
                    std::cerr << "AIToolbox: Input policy data is not sorted by state and action.\n";
                    fail = true;
                }
                if ( fail ) {
                    is.setstate(std::ios::failbit);
                    return is;
                }
            }
        }
        // Read succeeded
        for ( size_t s = 0; s < S; ++s ) {
            // Sanitization: Assign and normalize everything.
            p.setStatePolicy(s, policy.policy_[s]);
        }

        return is;
    }
}
