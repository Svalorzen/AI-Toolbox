#include <AIToolbox/MDP/RLModel.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox {
    namespace MDP {
        RLModel::RLModel( const Experience & exp, bool toSync ) : S(exp.getS()), A(exp.getA()), experience_(exp), transitions_(boost::extents[S][S][A]), rewards_(boost::extents[S][S][A]),
                                                       rand_(Impl::Seeder::getSeed()), sampleDistribution_(0.0, 1.0)
        {
            // Boost initializes everything to 0 automatically (uses default
            // element constructors).
            if ( toSync ) {
                sync();
                // Sync does not touch state-action pairs which have never been
                // seen. To keep the model consistent we set all of them as
                // self-absorbing. 
                for ( size_t s = 0; s < S; ++s )
                    for ( size_t a = 0; a < A; ++a )
                        if ( experience_.getVisitsSum(s, a) == 0ul )
                            transitions_[s][s][a] = 1.0;
            }
            else {
                // Make transition table true probability
                for ( size_t s = 0; s < S; ++s )
                    for ( size_t a = 0; a < A; ++a )
                        transitions_[s][s][a] = 1.0;
            }
        }

        void RLModel::sync() {
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    sync(s,a);
        }

        void RLModel::sync(size_t s, size_t a) {
            // Nothing to do
            unsigned long visitSum = experience_.getVisitsSum(s, a);
            if ( visitSum == 0ul ) return;
            // Normalize
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                unsigned long visits = experience_.getVisits(s, s1, a);
                // Normalize action reward over transition visits
                if ( visits != 0 ) {
                    rewards_[s][s1][a] = experience_.getReward(s, s1, a) / visits;
                }
                transitions_[s][s1][a] = static_cast<double>(visits) / static_cast<double>(visitSum);
            }
        }

        std::pair<size_t, double> RLModel::sample(size_t s, size_t a) const {
            double p = sampleDistribution_(rand_);

            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                if ( transitions_[s][s1][a] > p ) return std::make_pair(s1, rewards_[s][s1][a]);
                p -= transitions_[s][s1][a];
            }
            throw std::runtime_error("RLModel could not sample");
        }

        double RLModel::getTransitionProbability(size_t s, size_t s1, size_t a) const {
            return transitions_[s][s1][a];
        }

        double RLModel::getExpectedReward(size_t s, size_t s1, size_t a) const {
            return rewards_[s][s1][a];
        }

        size_t RLModel::getS() const { return S; }
        size_t RLModel::getA() const { return A; }
        const Experience & RLModel::getExperience() const { return experience_; }

        const RLModel::TransitionTable & RLModel::getTransitionFunction() const { return transitions_; }
        const RLModel::RewardTable &     RLModel::getRewardFunction()     const { return rewards_; }
    }
}
