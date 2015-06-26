#include <AIToolbox/MDP/SparseModel.hpp>

namespace AIToolbox {
    namespace MDP {
        SparseModel::SparseModel(size_t s, size_t a, double discount) : S(s), A(a), discount_(discount), transitions_(A, SparseMatrix2D(S, S)), rewards_(A, SparseMatrix2D(S, S)),
                                                                        rand_(Impl::Seeder::getSeed())
        {
            // Make transition table true probability
            for ( size_t a = 0; a < A; ++a )
                for ( size_t s = 0; s < S; ++s )
                    transitions_[a].insert(s, s) = 1.0;
        }

        std::tuple<size_t, double> SparseModel::sampleSR(size_t s, size_t a) const {
            size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);

            return std::make_tuple(s1, getExpectedReward(s, a, s1));
        }

        double SparseModel::getTransitionProbability(size_t s, size_t a, size_t s1) const {
            return transitions_[a].coeff(s, s1);
        }

        double SparseModel::getExpectedReward(size_t s, size_t a, size_t s1) const {
            return rewards_[a].coeff(s, s1);
        }

        void SparseModel::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        bool SparseModel::isTerminal(size_t s) const {
            bool answer = true;
            for ( size_t a = 0; a < A; ++a ) {
                if ( !checkEqualSmall(1.0, getTransitionProbability(s, a, s)) ) {
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

        const SparseMatrix2D & SparseModel::getTransitionFunction(size_t a) const { return transitions_[a]; }
        const SparseMatrix2D & SparseModel::getRewardFunction(size_t a)     const { return rewards_[a]; }
    }
}
