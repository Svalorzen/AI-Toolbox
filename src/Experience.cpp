#include <MDPToolbox/Experience.hpp>

#include <fstream>
#include <array>

#include <iostream>
using std::cout;

namespace MDPToolbox {
    Experience::Experience(size_t Ss, size_t Aa) : S(Ss), A(Aa), visits_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A])
    {
        reset();
    }

    void Experience::update(size_t s, size_t s1, size_t a, double rew) {
        visits_[s][s1][a]++;
        rewards_[s][s1][a] += rew;
    }

    void Experience::reset() {
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    visits_[s][s1][a] = 0;
                    rewards_[s][s1][a] = 0;
                }
            }
        }
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
        // old version  if ( !(is >> exp.visits_[s][s1][0] >> exp.visits_[s][s1][1] >> exp.rewards_[s][s1][0] >> exp.rewards_[s][s1][1])) {
        size_t S = exp.getS();
        size_t A = exp.getA();

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    if ( !(is >> exp.visits_[s][s1][a] >> exp.rewards_[s][s1][a] )) {
                        exp.reset();
                        return is;
                    }
                    // Verification/Sanitization
                    // Ignoring input reward if no visits.
                    if ( exp.visits_[s][s1][a] == 0 )
                        exp.rewards_[s][s1][a] = 0.0;
                }
            }
        }
        return is;
    }

    std::ostream& operator<<(std::ostream& os, const Experience & exp) {
        size_t S = exp.getS();
        size_t A = exp.getA();

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    os << exp.getVisits()[s][s1][a] << " " << exp.getRewards()[s][s1][a] << " ";
                }
            }
            os << "\n";
        }
        return os;
    }
    }
