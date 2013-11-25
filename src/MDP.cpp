#include <AIToolbox/MDP.hpp>

#include <chrono>

namespace AIToolbox {
    MDP::MDP(size_t s, size_t a) : S(s), A(a), experience_(s,a), transitions_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A]),
                                       rand_(std::chrono::system_clock::now().time_since_epoch().count()), sampleDistribution_(0.0, 1.0)
    {
        update();
    }

    MDP::MDP(Experience exp) : S(exp.getS()), A(exp.getA()), experience_(exp), transitions_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A]) {
        update();
    }

    void MDP::update() {
        rewards_ = experience_.getRewards();

        for ( size_t s = 0; s < S; s++ )
            for ( size_t a = 0; a < A; a++ )
                update(s,a);
    }

    void MDP::update(size_t s, size_t a) {
        unsigned long actionSum = 0;
        for ( size_t s1 = 0; s1 < S; s1++ ) {
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

    std::tuple<size_t, double> MDP::sample(size_t s, size_t a) const {
        double p = sampleDistribution_(rand_);

        for ( size_t s1 = 0; s1 < S; s1++ ) {
            if ( transitions_[s][s1][a] > p ) return std::make_tuple(s1, rewards_[s][s1][a]);
            p -= transitions_[s][s1][a];
        }
        return std::make_tuple(S-1, rewards_[s][S-1][a]);
    }

    size_t MDP::getS() const { return S; }
    size_t MDP::getA() const { return A; }

    Experience &       MDP::getExperience()       { return experience_; }
    const Experience & MDP::getExperience() const { return experience_; }

    const MDP::TransitionTable & MDP::getTransitionFunction() const { return transitions_; }
    const MDP::RewardTable &     MDP::getRewardFunction()     const { return rewards_; }
}
