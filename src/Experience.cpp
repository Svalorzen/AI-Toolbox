#include <MDPToolbox/Experience.hpp>

#include <fstream>
#include <array>

#include <iostream>
using std::cout;

namespace MDPToolbox {
    Experience::Experience(size_t Ss, size_t Aa) : S(Ss), A(Aa), visits_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A])
    {
        // Clear table
        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    visits_[s][s1][a] = 0;
                    rewards_[s][s1][a] = 0;
                }
            }
        }
        isValid_ = true;
    }

    bool Experience::load(const std::string & filename) {
        std::ifstream file(filename.c_str(), std::ifstream::in);

        for ( size_t s = 0; s < S; s++ ) {
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    if ( !(file >> visits_[s][s1][a] >> rewards_[s][s1][a] )) {
                        // old version  if ( !(file >> std::get<visits>(table_[i][j][0]) >> std::get<visits>(table_[i][j][1]) >> std::get<reward>(table_[i][j][0]) >> std::get<reward>(table_[i][j][1]))) {
                        isValid_ = false;
                        return false;
                    }
                    }
                }
            }
            // Should we verify the data in some way?
            file.close();
            isValid_ = true;
            return true;
        }

        bool Experience::save(std::string filename) const {
            if ( !isValid_ ) return false;

            std::ofstream file(filename.c_str(), std::ofstream::out);
            for ( size_t s = 0; s < S; s++ ) {
                for ( size_t s1 = 0; s1 < S; s1++ ) {
                    for ( size_t a = 0; a < A; a++ ) {
                        file << visits_[s][s1][a] << " " << rewards_[s][s1][a] << " ";
                    }
                }
                file << "\n";
            }

            file.close();
            return true;
        }

        void Experience::update(size_t s, size_t s1, size_t a, double rew) {
            visits_[s][s1][a]++;
            rewards_[s][s1][a] += rew;
        }

        bool Experience::isValid() const {
            return isValid_;
        }

        std::tuple<MDPToolbox::MDP::TransitionTable, MDPToolbox::MDP::RewardTable> Experience::getMDP() const {
            MDPToolbox::MDP::TransitionTable P(boost::extents[S][S][A]); // Can't initialize it here, long -> double
            MDPToolbox::MDP::RewardTable R(rewards_);

            double actionSum[A];
            for ( size_t s = 0; s < S; s++ ) {
                for ( size_t a = 0; a < A; a++ ) {
                    actionSum[a] = 0.0;
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
    }
