#include <AIToolbox/MDP/Model.hpp>

namespace AIToolbox::MDP {
    Model::Model(NoCheck, const size_t s, const size_t a, TransitionMatrix && t, RewardMatrix && r, const double d) :
            S(s), A(a), discount_(d),
            transitions_(std::move(t)),
            rewards_(std::move(r)),
            rand_(Impl::Seeder::getSeed()) {}

    Model::Model(const size_t s, const size_t a, const double discount) :
            S(s), A(a), discount_(discount), transitions_(A, Matrix2D(S, S)),
            rewards_(S, A), rand_(Impl::Seeder::getSeed())
    {
        // Make transition matrix true probability
        for ( size_t a = 0; a < A; ++a )
            transitions_[a].setIdentity();

        rewards_.setZero();
    }

    void Model::setTransitionFunction(const Matrix3D & t) {
        // First we verify data, without modifying anything...
        for ( size_t a = 0; a < A; ++a ) {
            for ( size_t s = 0; s < S; ++s ) {
                if ( t[a].row(s).minCoeff() < 0.0 ||
                     !checkEqualSmall(1.0, t[a].row(s).sum()) )
                {
                    throw std::invalid_argument("Input transition matrix does not contain valid probabilities.");
                }
            }
        }
        // Then we copy.
        transitions_ = t;
    }

    void Model::setRewardFunction(const RewardMatrix & r) {
        rewards_ = r;
    }

    std::tuple<size_t, double> Model::sampleSR(const size_t s, const size_t a) const {
        size_t s1 = sampleProbability(S, transitions_[a].row(s), rand_);

        return std::make_tuple(s1, rewards_(s, a));
    }

    double Model::getTransitionProbability(const size_t s, const size_t a, const size_t s1) const {
        return transitions_[a](s, s1);
    }

    double Model::getExpectedReward(const size_t s, const size_t a, const size_t) const {
        return rewards_(s, a);
    }

    void Model::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    bool Model::isTerminal(const size_t s) const {
        for ( size_t a = 0; a < A; ++a )
            if ( !checkEqualSmall(1.0, transitions_[a](s, s)) )
                return false;
        return true;
    }

    size_t Model::getS() const { return S; }
    size_t Model::getA() const { return A; }
    double Model::getDiscount() const { return discount_; }

    const Model::TransitionMatrix & Model::getTransitionFunction() const { return transitions_; }
    const Model::RewardMatrix &     Model::getRewardFunction()     const { return rewards_; }

    const Matrix2D & Model::getTransitionFunction(const size_t a) const { return transitions_[a]; }
}
