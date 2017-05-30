#include <AIToolbox/MDP/Algorithms/SARSA.hpp>

namespace AIToolbox::MDP {
    SARSA::SARSA(const size_t ss, const size_t aa, const double discount, const double alpha) :
            S(ss), A(aa), q_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setLearningRate(alpha);
    }

    void SARSA::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const size_t a1, const double rew) {
        q_(s, a) += alpha_ * ( rew + discount_ * q_(s1, a1) - q_(s, a) );
    }

    void SARSA::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double SARSA::getLearningRate() const { return alpha_; }

    void SARSA::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double SARSA::getDiscount() const { return discount_; }

    size_t SARSA::getS() const { return S; }
    size_t SARSA::getA() const { return A; }

    const QFunction & SARSA::getQFunction() const { return q_; }
}
