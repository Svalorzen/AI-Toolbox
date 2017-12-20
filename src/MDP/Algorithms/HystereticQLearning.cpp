#include <AIToolbox/MDP/Algorithms/HystereticQLearning.hpp>

namespace AIToolbox::MDP {
    HystereticQLearning::HystereticQLearning(const size_t ss, const size_t aa, const double discount, const double alpha, const double beta) :
            S(ss), A(aa), discount_(discount), q_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setPositiveLearningRate(alpha);
        setNegativeLearningRate(beta);
    }

    void HystereticQLearning::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        const auto delta = rew + discount_ * q_.row(s1).maxCoeff() - q_(s, a);
        if (delta >= 0)
            q_(s, a) += alpha_ * delta;
        else
            q_(s, a) += beta_ * delta;
    }

    void HystereticQLearning::setPositiveLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Positive learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double HystereticQLearning::getPositiveLearningRate() const { return alpha_; }

    void HystereticQLearning::setNegativeLearningRate(const double b) {
        if ( b < 0.0 || b > 1.0 ) throw std::invalid_argument("Negative learning rate parameter must be in [0,1]");
        beta_ = b;
    }

    double HystereticQLearning::getNegativeLearningRate() const { return beta_; }

    void HystereticQLearning::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double HystereticQLearning::getDiscount() const { return discount_; }

    size_t HystereticQLearning::getS() const { return S; }
    size_t HystereticQLearning::getA() const { return A; }

    const QFunction & HystereticQLearning::getQFunction() const { return q_; }
}
