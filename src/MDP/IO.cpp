#include <AIToolbox/MDP/IO.hpp>

#include <AIToolbox/PolicyInterface.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/SparseExperience.hpp>
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

        // Experience reader
        std::istream& operator>>(std::istream &is, Experience & exp) {
            const size_t S = exp.getS();
            const size_t A = exp.getA();

            Experience e(S,A);

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    long vSum = 0;
                    double rSum = 0.0;
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

                        vSum += e.visits_[s][a][s1];
                        rSum += e.rewards_[s][a][s1];
                    }
                    e.visitsSum_[s][a] = vSum;
                    e.rewardsSum_[s][a] = rSum;
                }
            }
            // This guarantees that if input is invalid we still keep the old Exp.
            exp = std::move(e);

            return is;
        }

        // SparseExperience reader
        std::istream& operator>>(std::istream &is, SparseExperience & exp) {
            const size_t S = exp.getS();
            const size_t A = exp.getA();

            long l;
            double d;

            SparseExperience e(S,A);

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    long vSum = 0;
                    double rSum = 0.0;
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        if ( !(is >> l >> d) ) {
                            std::cerr << "AIToolbox: Could not read Experience data.\n";
                            is.setstate(std::ios::failbit);
                            return is;
                        }
                        e.visits_[a].insert(s, s1) = l;
                        vSum += l;

                        // Verification/Sanitization
                        // Ignoring input reward if no visits.
                        if ( l > 0 && checkDifferentSmall(0.0, d) ) {
                            e.rewards_[a].insert(s, s1) = d;
                            rSum += d;
                        }
                    }
                    if ( vSum > 0 ) {
                        e.visitsSum_.insert(s, a) = vSum;
                        if ( checkDifferentSmall(0.0, rSum) ) e.rewardsSum_.insert(s, a) = rSum;
                    }
                }
            }
            // This guarantees that if input is invalid we still keep the old Exp.
            exp = std::move(e);

            return is;
        }

        // MDP::Model reader
        std::istream& operator>>(std::istream &is, Model & m) {
            const size_t S = m.getS();
            const size_t A = m.getA();

            Model in(S,A);

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    double sum = 0.0;
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        if ( !(is >> in.transitions_[a](s, s1) >> in.rewards_[a](s, s1) )) {
                            std::cerr << "AIToolbox: Could not read Model data.\n";
                            is.setstate(std::ios::failbit);
                            return is;
                        }
                        sum += in.transitions_[a](s, s1);
                    }

                    // Verification/Sanitization
                    if ( checkDifferentSmall(sum, 0.0) )
                        in.transitions_[a].row(s) /= sum;
                    else
                        in.transitions_[a](s, s) = 1.0;
                }
            }
            // This guarantees that if input is invalid we still keep the old Model.
            m = std::move(in);

            return is;
        }

        // MDP::SparseModel reader
        std::istream& operator>>(std::istream &is, SparseModel & m) {
            const size_t S = m.getS();
            const size_t A = m.getA();

            SparseModel in(S,A);
            double p, r;

            for ( size_t s = 0; s < S; ++s ) {
                for ( size_t a = 0; a < A; ++a ) {
                    double sum = 0.0;
                    for ( size_t s1 = 0; s1 < S; ++s1 ) {
                        if ( !(is >> p >> r )) {
                            std::cerr << "AIToolbox: Could not read Model data.\n";
                            is.setstate(std::ios::failbit);
                            return is;
                        }
                        else {
                            if ( checkDifferentSmall(0.0, p) ) {
                                sum += p;
                                in.transitions_[a].coeffRef(s, s1) = p;
                            }
                            if ( checkDifferentSmall(0.0, r) )
                                in.rewards_[a].coeffRef(s, s1) = r;
                        }
                    }
                    if ( checkDifferentSmall(sum, 0.0) )
                        in.transitions_[a].row(s) /= sum;
                    else
                        in.transitions_[a].coeffRef(s, s) = 1.0;
                }
            }
            // This guarantees that if input is invalid we still keep the old Model.
            m = std::move(in);

            return is;
        }

        // MDP::Policy reader
        std::istream& operator>>(std::istream &is, Policy &p) {
            const size_t S = p.getS();
            const size_t A = p.getA();

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
