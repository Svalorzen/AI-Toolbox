#include <AIToolbox/MDP/Algorithms/DoubleQLearning.hpp>

namespace AIToolbox::MDP {
    DoubleQLearning::DoubleQLearning(const size_t ss, const size_t aa, const double discount, const double alpha) :
            S(ss), A(aa), discount_(discount), dist_(0.5),
            qa_(makeQFunction(S, A)),
            qc_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setLearningRate(alpha);
    }

    void DoubleQLearning::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        size_t a1;

        if (dist_(rand_)) {
            qa_.row(s1).maxCoeff(&a1);
            const double change = alpha_ * ( rew + discount_ * (qc_(s1, a1) - qa_(s1, a1)) - qa_(s, a) );
            qa_(s, a) += change;
            qc_(s, a) += change;
        } else {
            (qc_.row(s1) - qa_.row(s1)).maxCoeff(&a1);
            qc_(s, a) += alpha_ * ( rew + discount_ * qa_(s1, a1) - (qc_(s, a) - qa_(s, a)) );
        }
    }

    void DoubleQLearning::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double DoubleQLearning::getLearningRate() const { return alpha_; }

    void DoubleQLearning::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double DoubleQLearning::getDiscount() const { return discount_; }

    size_t DoubleQLearning::getS() const { return S; }
    size_t DoubleQLearning::getA() const { return A; }

    const QFunction & DoubleQLearning::getQFunction() const { return qc_; }
    const QFunction & DoubleQLearning::getQFunctionA() const { return qa_; }
    QFunction DoubleQLearning::getQFunctionB() const { return qc_ - qa_; }
    void DoubleQLearning::setQFunction(const QFunction & qfun) {
        assert(qc_.rows() == qfun.rows());
        assert(qc_.cols() == qfun.cols());
        qa_ = qfun;
        qc_ = qfun * 2;
    }
}
