#include <AIToolbox/MDP/Model.hpp>

#include "Seeder.hpp"

namespace AIToolbox {
    namespace MDP {
        Model::Model(size_t s, size_t a) : S(s), A(a), transitions_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A]),
                                                       rand_(Impl::Seeder::getSeed()), sampleDistribution_(0.0, 1.0) 
        {
            std::fill(transitions_.data(), transitions_.data() + transitions_.num_elements(), 0.0);
            // Make transition table true probability
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    transitions_[s][s][a] = 1.0;
            std::fill(rewards_.data(), rewards_.data() + rewards_.num_elements(), 0.0);
        }

        std::pair<size_t, double> Model::sample(size_t s, size_t a) const {
            double p = sampleDistribution_(rand_);

            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                if ( transitions_[s][s1][a] > p ) return std::make_pair(s1, rewards_[s][s1][a]);
                p -= transitions_[s][s1][a];
            }
            return std::make_pair(S-1, rewards_[s][S-1][a]);
        }

        double Model::getTransitionProbability(size_t s, size_t s1, size_t a) const {
            return transitions_[s][s1][a];
        }

        double Model::getExpectedReward(size_t s, size_t s1, size_t a) const {
            return rewards_[s][s1][a];
        }

        size_t Model::getS() const { return S; }
        size_t Model::getA() const { return A; }

        const Model::TransitionTable & Model::getTransitionFunction() const { return transitions_; }
        const Model::RewardTable &     Model::getRewardFunction()     const { return rewards_; }
    }
}
