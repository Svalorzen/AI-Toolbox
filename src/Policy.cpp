#include <AIToolbox/Policy.hpp>

#include <chrono>
#include <algorithm>
#include <iostream>

namespace AIToolbox {
    Policy::Policy(size_t s, size_t a) : PolicyInterface(s, a), policy_(boost::extents[S][A])
    {
        // Random policy is default
        std::fill(policy_.data(), policy_.data() + policy_.num_elements(), 1.0/getA());
    }

    size_t Policy::sampleAction(size_t s) const {
        double p = sampleDistribution_(rand_);
        for ( size_t a = 0; a < A; a++ ) {
            if ( policy_[s][a] > p ) return a;
            p -= policy_[s][a];
        }
        // Return last action just in case
        return A-1;
    }

    double Policy::getActionProbability(size_t s, size_t a) const {
        return policy_[s][a];
    }

    std::vector<double> Policy::getStatePolicy( size_t s ) const {
        std::vector<double> statePolicy(A);

        std::copy(std::begin(policy_[s]), std::end(policy_[s]), std::begin(statePolicy));

        return statePolicy;
    }

    void Policy::setStatePolicy(size_t s, size_t a) {
        for ( size_t ax = 0; ax < A; ax++ )
            policy_[s][ax] = static_cast<double>( ax == a );
    }

    const Policy::PolicyTable & Policy::getPolicy() const {
        return policy_;
    }

    void Policy::prettyPrint(std::ostream & os) const {
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                if ( policy_[s][a] )
                    os << s << "\t" << a << "\t" << std::fixed << policy_[s][a] << "\n";
            }
        }
    }

    std::ostream& operator<<(std::ostream &os, const Policy &p) {
        size_t S = p.getS();
        size_t A = p.getA();

        auto & policy = p.getPolicy();

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                os << s << "\t" << a << "\t" << std::fixed << policy[s][a] << "\n";
            }
        }
        return os;
    }

    std::istream& operator>>(std::istream &is, Policy &p) {
        size_t S = p.getS();
        size_t A = p.getA();

        Policy policy(S, A);

        size_t scheck, acheck;

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                if ( ! ( is >> scheck >> acheck >> policy.policy_[s][a] ) ) {
                    std::cerr << "AIToolbox: Could not read policy data.\n";
                    return is;
                }
                else if ( policy.policy_[s][a] < 0.0 || policy.policy_[s][a] > 1.0 ) {
                    std::cerr << "AIToolbox: Input policy data contains non-probability values.\n";
                    return is;
                }
                else if ( scheck != s || acheck != a ) {
                    std::cerr << "AIToolbox: Input policy data is not sorted by state and action.\n";
                    return is;
                }
            }
        }
        // Read succeeded
        for ( size_t s = 0; s < S; s++ ) {
            // Sanitization: Assign and normalize everything.
            p.setStatePolicy(s, policy.policy_[s]);
        }

        return is;
    }
}
