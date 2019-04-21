#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

namespace AIToolbox::MDP {
    QLearning::QLearning(const size_t ss, const size_t aa, const double discount, const double alpha) :
            S(ss), A(aa), discount_(discount), q_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setLearningRate(alpha);
    }

    void QLearning::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        q_(s, a) += alpha_ * ( rew + discount_ * q_.row(s1).maxCoeff() - q_(s, a) );
    }

    void QLearning::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double QLearning::getLearningRate() const { return alpha_; }

    void QLearning::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double QLearning::getDiscount() const { return discount_; }

    size_t QLearning::getS() const { return S; }
    size_t QLearning::getA() const { return A; }

    const QFunction & QLearning::getQFunction() const { return q_; }
    void QLearning::setQFunction(const QFunction & qfun) { 
        assert(q_.rows() == qfun.rows());
        assert(q_.cols() == qfun.cols());
        q_ = qfun;
    }
}
