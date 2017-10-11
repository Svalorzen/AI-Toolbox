#include <AIToolbox/MDP/Algorithms/ExpectedSARSA.hpp>

namespace AIToolbox::MDP {
    ExpectedSARSA::ExpectedSARSA(QFunction & qfun, const PolicyInterface & policy, const double discount, const double alpha) :
            policy_(policy), S(policy_.getS()), A(policy_.getA()), q_(qfun)
    {
        setDiscount(discount);
        setLearningRate(alpha);
    }

    void ExpectedSARSA::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        double expectedQ = 0.0;
        for (size_t ai = 0; ai < A; ++ai)
            expectedQ += policy_.getActionProbability(s1, ai) * q_(s1, ai);

        q_(s, a) += alpha_ * ( rew + discount_ * expectedQ - q_(s, a) );
    }

    void ExpectedSARSA::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double ExpectedSARSA::getLearningRate() const { return alpha_; }

    void ExpectedSARSA::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double ExpectedSARSA::getDiscount() const { return discount_; }

    size_t ExpectedSARSA::getS() const { return S; }
    size_t ExpectedSARSA::getA() const { return A; }

    const QFunction & ExpectedSARSA::getQFunction() const { return q_; }
    const PolicyInterface & ExpectedSARSA::getPolicy() const { return policy_; }
}
