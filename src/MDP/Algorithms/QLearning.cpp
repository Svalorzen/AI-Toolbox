#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

namespace AIToolbox {
    namespace MDP {
        QLearning::QLearning(size_t ss, size_t aa, double discount, double alpha) : S(ss), A(aa), alpha_(alpha), discount_(discount), q_(makeQFunction(S, A)) {
            if ( discount_ <= 0.0 || alpha_ > 1.0 )        throw std::invalid_argument("Discount parameter must be in (0,1]");
            if ( alpha_ <= 0.0 || alpha_ > 1.0 )        throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        }

        void QLearning::stepUpdateQ(size_t s, size_t a, size_t s1, double rew) {
            q_(s, a) += alpha_ * ( rew + discount_ * q_.row(s1).maxCoeff() - q_(s, a) );
        }

        void QLearning::setLearningRate(double a) {
            if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            alpha_ = a;
        }

        double QLearning::getLearningRate() const { return alpha_; }

        void QLearning::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        double QLearning::getDiscount() const { return discount_; }

        size_t QLearning::getS() const { return S; }
        size_t QLearning::getA() const { return A; }

        const QFunction & QLearning::getQFunction() const { return q_; }
    }
}
