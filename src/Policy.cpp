#include <AIToolbox/Policy.hpp>

#include <chrono>
#include <algorithm>
#include <iostream>

namespace AIToolbox {
    Policy::Policy(size_t sNum, size_t aNum) : S(sNum), A(aNum), policy_(boost::extents[S][A]),
                                               rand_(std::chrono::system_clock::now().time_since_epoch().count()), sampleDistribution_(0.0, 1.0), randomDistribution_(0, A-1)
    {
        // Random policy is default
        std::fill(policy_.data(), policy_.data() + policy_.num_elements(), 1.0/A);
    }

    size_t Policy::getAction(size_t s, double epsilon) const {
        if ( epsilon < 1.0 ) {
            double greedy = sampleDistribution_(rand_);
            if ( greedy > epsilon ) {
                // RANDOM!
                return randomDistribution_(rand_);
            }
        }
        // GREEDY!
        double p = sampleDistribution_(rand_);
        for ( size_t a = 0; a < A; a++ ) {
            if ( policy_[s][a] > p ) return s;
            p -= policy_[s][a];
        }
        return S-1+epsilon;
    }

    std::vector<double> Policy::getStatePolicy( size_t s ) const {
        std::vector<double> statePolicy(A);

        std::copy(std::begin(policy_[s]), std::end(policy_[s]), std::begin(statePolicy));

        return statePolicy;
    }

    void Policy::setPolicy(size_t s, size_t a) {
        for ( size_t ax = 0; ax < A; ax++ )
            policy_[s][ax] = static_cast<double>( ax == a );
    }

    size_t Policy::getS() const {
        return S;
    }

    size_t Policy::getA() const {
        return A;
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
        for ( size_t s = 0; s < S; s++ )
            // Sanitization: Assign and normalize everything.
            p.setPolicy(s, policy.policy_[s]);

        return is;
    }
}
