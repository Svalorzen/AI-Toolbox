#include <AIToolbox/MDP/SparseModel.hpp>

namespace AIToolbox {
    namespace MDP {
        SparseModel::SparseModel(size_t s, size_t a, double discount) : S(s), A(a), discount_(discount), rand_(Impl::Seeder::getSeed())
        {
            // Make transition table true probability
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    transitions_.set(1.0, s, a, s);
        }

        std::tuple<size_t, double> SparseModel::sampleSR(size_t s, size_t a) const {
            size_t s1 = sampleProbability(S, transitions_.getRow(S, s, a), rand_);

            return std::make_tuple(s1, rewards_(s, a, s1));
        }

        double SparseModel::getTransitionProbability(size_t s, size_t a, size_t s1) const {
            return transitions_(s, a, s1);
        }

        double SparseModel::getExpectedReward(size_t s, size_t a, size_t s1) const {
            return rewards_(s, a, s1);
        }

        void SparseModel::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        bool SparseModel::isTerminal(size_t s) const {
            bool answer = true;
            for ( size_t a = 0; a < A; ++a ) {
                if ( !checkEqualSmall(1.0, transitions_(s, a, s)) ) {
                    answer = false;
                    break;
                }
            }
            return answer;
        }

        size_t SparseModel::getS() const { return S; }
        size_t SparseModel::getA() const { return A; }
        double SparseModel::getDiscount() const { return discount_; }

        const SparseModel::TransitionTable & SparseModel::getTransitionFunction() const { return transitions_; }
        const SparseModel::RewardTable &     SparseModel::getRewardFunction()     const { return rewards_; }
    }
}
