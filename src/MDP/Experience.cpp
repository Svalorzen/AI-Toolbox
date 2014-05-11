#include <AIToolbox/MDP/Experience.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace AIToolbox {
    namespace MDP {
        Experience::Experience(size_t s, size_t a) : S(s), A(a), visits_(boost::extents[S][A][S]), visitsSum_(boost::extents[S][A]),
        rewards_(boost::extents[S][A][S]), rewardsSum_(boost::extents[S][A]) {}

        void Experience::record(size_t s, size_t a, size_t s1, double rew) {
            visits_[s][a][s1]   += 1;
            visitsSum_[s][a]    += 1;

            rewards_[s][a][s1]  += rew;
            rewardsSum_[s][a]   += rew;
        }

        void Experience::reset() {
            std::fill(visits_.data(), visits_.data() + visits_.num_elements(), 0ul);
            std::fill(visitsSum_.data(), visitsSum_.data() + visitsSum_.num_elements(), 0ul);

            std::fill(rewards_.data(), rewards_.data() + rewards_.num_elements(), 0.0);
            std::fill(rewardsSum_.data(), rewardsSum_.data() + rewardsSum_.num_elements(), 0.0);
        }

        unsigned long Experience::getVisits(size_t s, size_t a, size_t s1) const {
            return visits_[s][a][s1];
        }

        unsigned long Experience::getVisitsSum(size_t s, size_t a) const {
            return visitsSum_[s][a];
        }

        double Experience::getReward(size_t s, size_t a, size_t s1) const {
            return rewards_[s][a][s1];
        }

        double Experience::getRewardSum(size_t s, size_t a) const {
            return rewardsSum_[s][a];
        }

        const Experience::VisitTable & Experience::getVisitTable() const {
            return visits_;
        }

        const Experience::RewardTable & Experience::getRewardTable() const {
            return rewards_;
        }

        size_t Experience::getS() const {
            return S;
        }

        size_t Experience::getA() const {
            return A;
        }

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
            exp = e;

            return is;
        }

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
    }
}
