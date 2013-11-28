#include <AIToolbox/MDP/RLModel.hpp>

namespace AIToolbox {
    namespace MDP {
        RLModel::RLModel( const Experience & exp ) : Model(exp.getS(), exp.getA()), experience_(exp) {
            std::fill(transitions_.data(), transitions_.data() + transitions_.num_elements(), 0.0);
            // Make transition table true probability
            for ( size_t s = 0; s < S; s++ )
                transitions_[s][s][0] = 1.0;
            std::fill(rewards_.data(), rewards_.data() + rewards_.num_elements(), 0.0);
        }

        void RLModel::sync() {
            for ( size_t s = 0; s < S; s++ )
                for ( size_t a = 0; a < A; a++ )
                    sync(s,a);
        }

        void RLModel::sync(size_t s, size_t a) {
            unsigned long actionSum = 0;
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                rewards_[s][s1][a] = experience_.getRewards()[s][s1][a];

                unsigned temp = experience_.getVisits()[s][s1][a];
                transitions_[s][s1][a] = static_cast<double>(temp);
                // actionSum contains the numer of times we have executed action 'a' in state 's'
                actionSum += temp;
            }
            // Normalize
            for ( size_t s1 = 0; s1 < S; s1++ ) {
                // If we never executed 'a' during 'i'
                if ( actionSum == 0.0 ) {
                    // Create shadow state since we never encountered it
                    if ( s == s1 )
                        transitions_[s][s1][a] = 1.0;
                    else
                        transitions_[s][s1][a] = 0.0;
                    // Reward is already 0 anyway
                }
                else {
                    // Normalize action reward over transition visits
                    if ( transitions_[s][s1][a] != 0.0 ) {
                       rewards_[s][s1][a] /= transitions_[s][s1][a];
                    }
                    // Normalize transition probability (times we went to 's1' / times we executed 'a' in 's'
                    transitions_[s][s1][a] /= actionSum;
                }
            }
        }

        const Experience & RLModel::getExperience() const { return experience_; }
    }
}
