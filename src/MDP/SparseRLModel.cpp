#include <AIToolbox/MDP/SparseRLModel.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        SparseRLModel::SparseRLModel( const Experience & exp, double discount, bool toSync ) : S(exp.getS()), A(exp.getA()), experience_(exp), transitions_(A, SparseMatrix2D(S, S)), rewards_(A, SparseMatrix2D(S, S)),
                                                       rand_(Impl::Seeder::getSeed())
        {
            setDiscount(discount);

            if ( toSync ) {
                sync();
                // Sync does not touch state-action pairs which have never been
                // seen. To keep the model consistent we set all of them as
                // self-absorbing.
                for ( size_t a = 0; a < A; ++a )
                    for ( size_t s = 0; s < S; ++s )
                        if ( experience_.getVisitsSum(s, a) == 0ul )
                            transitions_[a].insert(s, s) = 1.0;
            }
            else {
                // Make transition table true probability
                for ( size_t a = 0; a < A; ++a )
                    transitions_[a].setIdentity();
            }
        }

        void SparseRLModel::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        void SparseRLModel::sync() {
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s = 0; s < S; ++s )
                    sync(s,a);
        }

        void SparseRLModel::sync(size_t s, size_t a) {
            // Nothing to do
            unsigned long visitSum = experience_.getVisitsSum(s, a);
            if ( visitSum == 0ul ) return;

            // Create reciprocal for fast division
            double visitSumReciprocal = 1.0 / visitSum;

            // Normalize
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                unsigned long visits = experience_.getVisits(s, a, s1);
                // Normalize action reward over transition visits
                if ( visits != 0 ) {
                    rewards_[a].coeffRef(s, s1) = experience_.getReward(s, a, s1) / visits;
                }
                transitions_[a].coeffRef(s, s1) = static_cast<double>(visits) * visitSumReciprocal;
            }
        }

        void SparseRLModel::sync(size_t s, size_t a, size_t s1) {
            unsigned long visitSum = experience_.getVisitsSum(s, a);
            // This function will not work on the first sync due to a division
            // by zero (and possibly because the vector will not be empty, and
            // even then then the computed new sum would be wrong). The second
            // condition is related to numerical errors. Once in a while we reset
            // those by forcing a true update using real data.
            if ( visitSum < 2ul || !(visitSum % 10000ul) ) return sync(s, a);

            double newVisits = static_cast<double>(experience_.getVisits(s, a, s1));

            // Update reward for this transition (all others stay the same).
            rewards_[a].coeffRef(s, s1) = experience_.getReward(s, a, s1) / newVisits;

            double newTransitionValue = newVisits / static_cast<double>(visitSum - 1);
            double newVectorSum = 1.0 + (newTransitionValue - transitions_[a].coeff(s, s1));
            // This works because as long as all the values in the transition have the same denominator
            // (in this case visitSum-1), then the numerators do not matter, as we can simply normalize.
            // In the end of the process the new values will be the same as if we updated directly using
            // an increased denominator, and thus we will be able to call this function again correctly.
            transitions_[a].coeffRef(s, s1) = newTransitionValue;
            transitions_[a].row(s) /= newVectorSum;
        }

        std::tuple<size_t, double> SparseRLModel::sampleSR(size_t s, size_t a) const {
            size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);

            return std::make_tuple(s1, rewards_[a].coeff(s, s1));
        }

        double SparseRLModel::getTransitionProbability(size_t s, size_t a, size_t s1) const {
            return transitions_[a].coeff(s, s1);
        }

        double SparseRLModel::getExpectedReward(size_t s, size_t a, size_t s1) const {
            return rewards_[a].coeff(s, s1);
        }

        bool SparseRLModel::isTerminal(size_t s) const {
            bool answer = true;
            for ( size_t a = 0; a < A; ++a ) {
                if ( !checkEqualSmall(1.0, transitions_[a].coeff(s, s)) ) {
                    answer = false;
                    break;
                }
            }
            return answer;
        }

        size_t SparseRLModel::getS() const { return S; }
        size_t SparseRLModel::getA() const { return A; }
        double SparseRLModel::getDiscount() const { return discount_; }
        const Experience & SparseRLModel::getExperience() const { return experience_; }

        const SparseRLModel::TransitionTable & SparseRLModel::getTransitionFunction() const { return transitions_; }
        const SparseRLModel::RewardTable &     SparseRLModel::getRewardFunction()     const { return rewards_; }

        const SparseMatrix2D & SparseRLModel::getTransitionFunction(size_t a) const { return transitions_[a]; }
        const SparseMatrix2D & SparseRLModel::getRewardFunction(size_t a)     const { return rewards_[a]; }
    }
}
