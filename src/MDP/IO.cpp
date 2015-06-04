#include <AIToolbox/MDP/IO.hpp>

#include <AIToolbox/PolicyInterface.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Policies/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        // Global discrete policy writer
        std::ostream& operator<<(std::ostream &os, const PolicyInterface<size_t> & p) {
            size_t S = p.getS();
            size_t A = p.getA();

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    os << s << '\t' << a << '\t' << std::fixed << p.getActionProbability(s,a) << '\n';
                }
            }
            return os;
        }

        // Experience writer
        std::ostream& operator<<(std::ostream& os, const Experience & exp) {
            size_t S = exp.getS();
            size_t A = exp.getA();

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        os << exp.getVisits(s, a, s1) << '\t' << exp.getReward(s, a, s1) << '\t';
                    }
                }
                os << '\n';
            }
            return os;
        }

        // Experience reader
        std::istream& operator>>(std::istream &is, Experience & exp) {
            size_t S = exp.getS();
            size_t A = exp.getA();

            Experience e(S,A);

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        if ( !(is >> e.visits_[s][a][s1] >> e.rewards_[s][a][s1] )) {
                            std::cerr << "AIToolbox: Could not read Experience data.\n";
                            is.setstate(std::ios::failbit);
                            return is;
                        }
                        // Verification/Sanitization
                        // Ignoring input reward if no visits.
                        if ( e.visits_[s][a][s1] == 0 )
                            e.rewards_[s][a][s1] = 0.0;
                    }
                }
            }
            // This guarantees that if input is invalid we still keep the old Exp.
            std::swap(exp, e);

            return is;
        }

        // MDP::Model reader
        std::istream& operator>>(std::istream &is, Model & m) {
            size_t S = m.getS();
            size_t A = m.getA();

            Model in(S,A);

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        if ( !(is >> in.transitions_[s][a][s1] >> in.rewards_[s][a][s1] )) {
                            std::cerr << "AIToolbox: Could not read Model data.\n";
                            is.setstate(std::ios::failbit);
                            return is;
                        }
                    }
                    // Verification/Sanitization
                    auto ref = in.transitions_[s][a];
                    normalizeProbability(std::begin(ref), std::end(ref), std::begin(ref));
                }
            }
            // This guarantees that if input is invalid we still keep the old Model.
            std::swap(m, in);

            return is;
        }

        // MDP::SparseModel reader
        std::istream& operator>>(std::istream &is, SparseModel & m) {
            size_t S = m.getS();
            size_t A = m.getA();

            SparseModel in(S,A);
            double p, r;

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        if ( !(is >> p >> r )) {
                            std::cerr << "AIToolbox: Could not read Model data.\n";
                            is.setstate(std::ios::failbit);
                            return is;
                        }
                        else {
                            if ( checkDifferentSmall(0.0, p) ) in.transitions_.set(p, s, a, s1);
                            if ( checkDifferentSmall(0.0, r) ) in.rewards_.set(r, s, a, s1);
                        }
                    }
                    // Verification/Sanitization
                    auto ref = in.transitions_.getRow(S, s, a);
                    normalizeProbability(std::begin(ref), std::end(ref), std::begin(ref));
                    for ( size_t s1 = 0; s1 < S; ++s1 )
                        in.transitions_.set(ref[s1], s, a, s1);
                }
            }
            // This guarantees that if input is invalid we still keep the old Model.
            std::swap(m, in);

            return is;
        }

        // MDP::Policy reader
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
}
