#include <AIToolbox/Experience.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace AIToolbox {
    Experience::Experience(size_t s, size_t a) : S(s), A(a), visits_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A])
    {
        reset();
    }

    void Experience::record(size_t s, size_t s1, size_t a, double rew) {
        visits_[s][s1][a]  += 1;
        rewards_[s][s1][a] += rew;
    }

    void Experience::reset() {
        std::fill(visits_.data(), visits_.data() + visits_.num_elements(), 0ul);
        std::fill(rewards_.data(), rewards_.data() + rewards_.num_elements(), 0.0);
    }

    unsigned long Experience::getVisits(size_t s, size_t s1, size_t a) const {
        return visits_[s][s1][a];
    }

    double Experience::getReward(size_t s, size_t s1, size_t a) const {
        return rewards_[s][s1][a];
    }

    const Experience::VisitTable & Experience::getVisits() const {
        return visits_;
    }

    const Experience::RewardTable & Experience::getRewards() const {
        return rewards_;
    }

    size_t Experience::getS() const {
        return S;
    }

    size_t Experience::getA() const {
        return A;
    }

    std::istream& operator>>(std::istream &is, Experience & exp) {
        // old version  if ( !(is >> exp.visits_[s][s1][0] >> exp.visits_[s][s1][1] >> exp.rewards_[s][s1][0] >> exp.rewards_[s][s1][1]))
        size_t S = exp.getS();
        size_t A = exp.getA();

        Experience e(S,A);

        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                for ( size_t a = 0; a < A; ++a ) {
                    if ( !(is >> e.visits_[s][s1][a] >> e.rewards_[s][s1][a] )) {
                        std::cerr << "AIToolbox: Could not read Experience data.\n";
                        is.setstate(std::ios::failbit);
                        return is;
                    }
                    // Verification/Sanitization
                    // Ignoring input reward if no visits.
                    if ( e.visits_[s][s1][a] == 0 )
                        e.rewards_[s][s1][a] = 0.0;
                }
            }
        }
        // This guarantees that if input fucks up we still keep the old Exp.
        exp = e;

        return is;
    }

    std::ostream& operator<<(std::ostream& os, const Experience & exp) {
        size_t S = exp.getS();
        size_t A = exp.getA();

        for ( size_t s = 0; s < S; ++s ) {
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                for ( size_t a = 0; a < A; ++a ) {
                    os << exp.getVisits(s, s1, a) << '\t' << exp.getReward(s, s1, a) << '\t';
                }
            }
            os << '\n';
        }
        return os;
    }
}
