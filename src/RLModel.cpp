#include <AIToolbox/MDP/RLModel.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        RLModel::RLModel( const Experience & exp, bool toSync ) : S(exp.getS()), A(exp.getA()), experience_(exp), transitions_(boost::extents[S][A][S]), rewards_(boost::extents[S][A][S]),
                                                       rand_(Impl::Seeder::getSeed())
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
                            transitions_[s][a][s] = 1.0;
            }
            else {
                // Make transition table true probability
                for ( size_t s = 0; s < S; ++s )
                    for ( size_t a = 0; a < A; ++a )
                        transitions_[s][a][s] = 1.0;
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
                unsigned long visits = experience_.getVisits(s, a, s1);
                // Normalize action reward over transition visits
                if ( visits != 0 ) {
                    rewards_[s][a][s1] = experience_.getReward(s, a, s1) / visits;
                }
                transitions_[s][a][s1] = static_cast<double>(visits) / static_cast<double>(visitSum);
            }
        }

        std::pair<size_t, double> RLModel::sample(size_t s, size_t a) const {
            size_t s1 = sampleProbability(transitions_[s][a], S, rand_);

            return std::make_pair(s1, rewards_[s][a][s1]);
        }

        double RLModel::getTransitionProbability(size_t s, size_t a, size_t s1) const {
            return transitions_[s][a][s1];
        }

        double RLModel::getExpectedReward(size_t s, size_t a, size_t s1) const {
            return rewards_[s][a][s1];
        }

        size_t RLModel::getS() const { return S; }
        size_t RLModel::getA() const { return A; }
        const Experience & RLModel::getExperience() const { return experience_; }

        const RLModel::TransitionTable & RLModel::getTransitionFunction() const { return transitions_; }
        const RLModel::RewardTable &     RLModel::getRewardFunction()     const { return rewards_; }
    }
}
