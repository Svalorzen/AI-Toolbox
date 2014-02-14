#include <AIToolbox/MDP/QLearning.hpp>

#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox {
    namespace MDP {
        QLearning::QLearning(size_t s, size_t a, double alpha, double discount) : S(s), A(a), alpha_(alpha), discount_(discount), q_(makeQFunction(S,A)) {
            if ( alpha_ <= 0 || alpha_ > 1 )        throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            if ( discount_ <= 0 || discount_ > 1 )  throw std::invalid_argument("Discount parameter must be in (0,1]");
        }

        void QLearning::stepUpdateQ(size_t s, size_t s1, size_t a, double rew) {
            q_[s][a] += alpha_ * ( rew + discount_ * (*std::max_element(std::begin(q_[s1]),std::end(q_[s1]))) - q_[s][a] );
        }

        void QLearning::setLearningRate(double a) {
            if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            alpha_ = a;
        }

        double QLearning::getLearningRate() const {
            return alpha_;
        }

        void QLearning::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
            discount_ = d;
        }

        double QLearning::getDiscount() const {
            return discount_;
        }

        const QFunction & QLearning::getQFunction() const {
            return q_;
        }

        size_t QLearning::getS() const { return S; }
        size_t QLearning::getA() const { return A; }
    }
}
