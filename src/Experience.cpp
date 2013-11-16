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

    std::tuple<MDPToolbox::MDP::TransitionTable, MDPToolbox::MDP::RewardTable> Experience::getMDP() const {
        MDPToolbox::MDP::TransitionTable P(boost::extents[S][S][A]); // Can't initialize it here, long -> double
        MDPToolbox::MDP::RewardTable R(rewards_);

        std::vector<double> actionSum(A, 0.0);
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t a = 0; a < A; a++ ) {
                for ( size_t s1 = 0; s1 < S; s1++ ) {
                    P[s][s1][a] = static_cast<double>(visits_[s][s1][a]);
                    // actionSum contains the time we have executed action 'a' in state 's'
                    actionSum[a] += P[s][s1][a];
                }
                // Normalize
                for ( size_t s1 = 0; s1 < S; s1++ ) {
                    // If we never executed 'a' during 'i'
                    if ( actionSum[a] == 0.0 ) {
                        // Create shadow state since we never encountered it
                        if ( s == s1 )
                            P[s][s1][a] = 1.0;
                        else
                            P[s][s1][a] = 0.0;
                        // Reward is already 0 anyway
                    }
                    else {
                        // Normalize action reward over transition visits
                        if ( P[s][s1][a] != 0.0 ) {
                            R[s][s1][a] /= P[s][s1][a];
                        }
                        // Normalize transition probability (times we went to 's1' / times we executed 'a' in 's'
                        P[s][s1][a] /= actionSum[a];
                    }
                }
            }
        }
        return std::make_tuple(P,R);
    }

    long Experience::getVisits( size_t s, size_t s1, size_t a ) const {
        return visits_[s][s1][a];
    }

    double Experience::getReward( size_t s, size_t s1, size_t a ) const {
        return rewards_[s][s1][a];
    }

    size_t Experience::getS() const {
        return S;
    }

    size_t Experience::getA() const {
        return A;
    }

    std::istream& operator>>(std::istream &is, Experience & exp) {
        // old version  if ( !(file >> std::get<visits>(table_[i][j][0]) >> std::get<visits>(table_[i][j][1]) >> std::get<reward>(table_[i][j][0]) >> std::get<reward>(table_[i][j][1]))) {
        size_t S = exp.getS();
        size_t A = exp.getA();

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    if ( !(is >> exp.visits_[s][s1][a] >> exp.rewards_[s][s1][a] )) {
                        exp.reset();
                        return is;
                    }
                    // Should we verify the data in some way?
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
                    os << exp.getVisits(s, s1, a) << " " << exp.getReward(s, s1, a) << " ";
                }
            }
            os << "\n";
        }
        return os;
    }
    }
