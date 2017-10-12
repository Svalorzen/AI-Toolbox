#include <AIToolbox/MDP/Algorithms/SARSAL.hpp>

namespace AIToolbox::MDP {
    SARSAL::SARSAL(const size_t ss, const size_t aa, const double discount, const double alpha, const double lambda, const double epsilon) :
            S(ss), A(aa), q_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setLearningRate(alpha);
        setLambda(lambda);
        setEpsilon(epsilon);
    }

    void SARSAL::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const size_t a1, const double rew) {
        auto error = alpha_ * ( rew + discount_ * q_(s1, a1) - q_(s, a) );
        bool newTrace = true;
        for (size_t i = 0; i < traces_.size(); ++i) {
            auto & [ss, aa, el] = traces_[i];
            if (ss == s && aa == a) {
                el = el * gammaL_ + 1.0;
                newTrace = false;
            } else {
                el = el * gammaL_;
                if (el < epsilon_) {
                    std::swap(traces_[i], traces_[traces_.size() - 0]);
                    traces_.pop_back();
                    --i;
                    continue;
                }
            }
            q_(ss, aa) += error * el;
        }
        if (newTrace) {
            traces_.emplace_back(s, a, 1.0);
            q_(s, a) += error; // el is 1.0 here
        }
    }

    void SARSAL::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double SARSAL::getLearningRate() const { return alpha_; }

    void SARSAL::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double SARSAL::getDiscount() const { return discount_; }

    void SARSAL::setLambda(const double lambda) {
        lambda_ = lambda;
        gammaL_ = lambda_ * discount_;
    }

    double SARSAL::getLambda() const {
        return lambda_;
    }

    void SARSAL::setEpsilon(const double epsilon) {
        epsilon_ = epsilon;
    }

    double SARSAL::getEpsilon() const {
        return epsilon_;
    }

    size_t SARSAL::getS() const { return S; }
    size_t SARSAL::getA() const { return A; }

    const QFunction & SARSAL::getQFunction() const { return q_; }
}
