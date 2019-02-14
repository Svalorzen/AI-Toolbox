#include <AIToolbox/MDP/SparseModel.hpp>

namespace AIToolbox::MDP {
    SparseModel::SparseModel(NoCheck, const size_t s, const size_t a, TransitionMatrix && t, RewardMatrix && r, const double d) :
            S(s), A(a), discount_(d), transitions_(t), rewards_(r), rand_(Impl::Seeder::getSeed()) {}

    SparseModel::SparseModel(const size_t s, const size_t a, const double discount) :
            S(s), A(a), discount_(discount), transitions_(A, SparseMatrix2D(S, S)),
            rewards_(S, A), rand_(Impl::Seeder::getSeed())
    {
        // Make transition matrix true probability
        for ( size_t a = 0; a < A; ++a )
            transitions_[a].setIdentity();
    }

    void SparseModel::setTransitionFunction(const SparseMatrix3D & t) {
        // First we verify data, without modifying anything...
        for ( size_t a = 0; a < A; ++a ) {
            // Eigen sparse does not implement minCoeff so we can't check for negatives.
            // So we force the matrix to its abs, and if then the sum goes haywire then
            // we found an error.
            for ( size_t s = 0; s < S; ++s ) {
                if ( !checkEqualSmall(1.0, t[a].row(s).sum()) )
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
                if ( !checkEqualSmall(1.0, t[a].row(s).cwiseAbs().sum()) )
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
            }
        }
        // Then we copy.
        transitions_ = t;
    }

    void SparseModel::setRewardFunction(const RewardMatrix & r) {
        rewards_ = r;
    }

    std::tuple<size_t, double> SparseModel::sampleSR(const size_t s, const size_t a) const {
        const size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);

        return std::make_tuple(s1, getExpectedReward(s, a, s1));
    }

    double SparseModel::getTransitionProbability(const size_t s, const size_t a, const size_t s1) const {
        return transitions_[a].coeff(s, s1);
    }

    double SparseModel::getExpectedReward(const size_t s, const size_t a, const size_t) const {
        return rewards_.coeff(s, a);
    }

    void SparseModel::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    bool SparseModel::isTerminal(const size_t s) const {
        for ( size_t a = 0; a < A; ++a )
            if ( !checkEqualSmall(1.0, getTransitionProbability(s, a, s)) )
                return false;
        return true;
    }

    size_t SparseModel::getS() const { return S; }
    size_t SparseModel::getA() const { return A; }
    double SparseModel::getDiscount() const { return discount_; }

    const SparseModel::TransitionMatrix & SparseModel::getTransitionFunction() const { return transitions_; }
    const SparseModel::RewardMatrix &     SparseModel::getRewardFunction()     const { return rewards_; }

    const SparseMatrix2D & SparseModel::getTransitionFunction(const size_t a) const { return transitions_[a]; }
}
