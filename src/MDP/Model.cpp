#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        Model::Model(size_t s, size_t a) : S(s), A(a), transitions_(boost::extents[S][A][S]), rewards_(boost::extents[S][A][S]),
                                                       rand_(Impl::Seeder::getSeed())
        {
            // Make transition table true probability
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    transitions_[s][a][s] = 1.0;
        }

        std::pair<size_t, double> Model::sample(size_t s, size_t a) const {
            size_t s1 = sampleProbability(transitions_[s][a], S, rand_);

            return std::make_pair(s1, rewards_[s][a][s1]);
        }

        double Model::getTransitionProbability(size_t s, size_t a, size_t s1) const {
            return transitions_[s][a][s1];
        }

        double Model::getExpectedReward(size_t s, size_t a, size_t s1) const {
            return rewards_[s][a][s1];
        }

        size_t Model::getS() const { return S; }
        size_t Model::getA() const { return A; }

        const Model::TransitionTable & Model::getTransitionFunction() const { return transitions_; }
        const Model::RewardTable &     Model::getRewardFunction()     const { return rewards_; }
    }
}
