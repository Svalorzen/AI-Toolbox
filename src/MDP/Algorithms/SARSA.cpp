#include <AIToolbox/MDP/Algorithms/SARSA.hpp>

namespace AIToolbox {
    namespace MDP {
        SARSA::SARSA(size_t ss, size_t aa, double discount, double alpha) : S(ss), A(aa), alpha_(alpha), discount_(discount), q_(makeQFunction(S, A)) {
            if ( discount_ <= 0.0 || alpha_ > 1.0 )        throw std::invalid_argument("Discount parameter must be in (0,1]");
            if ( alpha_ <= 0.0 || alpha_ > 1.0 )        throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        }

        void SARSA::stepUpdateQ(size_t s, size_t a, size_t s1, size_t a1, double rew) {
            q_(s, a) += alpha_ * ( rew + discount_ * q_(s1, a1) - q_(s, a) );
        }

        void SARSA::setLearningRate(double a) {
            if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            alpha_ = a;
        }

        double SARSA::getLearningRate() const { return alpha_; }

        void SARSA::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        double SARSA::getDiscount() const { return discount_; }

        size_t SARSA::getS() const { return S; }
        size_t SARSA::getA() const { return A; }

        const QFunction & SARSA::getQFunction() const { return q_; }
    }
}
