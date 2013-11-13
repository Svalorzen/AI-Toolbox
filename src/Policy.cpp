#include <MDPToolbox/Policy.hpp>

#include <algorithm>

namespace MDPToolbox {
    Policy::Policy(size_t sNum, size_t aNum) : S(sNum), A(aNum) {
        policy_.resize(S);

        for ( size_t s = 0; s < S; s++ ) {
            policy_[s].resize(A);

            for ( size_t a = 0; a < A; a++ ) {
                // Random policy
                policy_[s][a] = 1.0/A;
            }
        }
    }

    Policy::StatePolicy Policy::getStatePolicy( size_t s ) const {
        return policy_.at(s);
    }

    void Policy::setPolicy(size_t s, const StatePolicy & apt) {
        if ( std::accumulate(begin(apt), end(apt), 0.0) != 1.0 )
            throw std::runtime_error("Policy values for a state must sum to one");

       policy_[s] = apt;
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

    std::ostream& operator<<(std::ostream &os, const Policy &p) {
        size_t S = p.getS();
        size_t A = p.getA();

        for ( size_t s = 0; s < S; s++ ) {
            auto policy = p.getStatePolicy(s);
            for ( size_t a = 0; a < A; a++ ) {
                if ( policy[a] )
                    os << s << " " << a << " " << policy[a] << "\n";
            }
        }
        return os;
    }
}
