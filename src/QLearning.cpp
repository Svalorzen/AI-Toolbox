#include <AIToolbox/MDP/QLearning.hpp>

#include <cassert>

namespace AIToolbox {
    namespace MDP {
         
        QLearning::QLearning(double alpha, double discount) : alpha_(alpha), discount_(discount) {
            if ( alpha_ <= 0 || alpha_ > 1 )  throw std::runtime_error("Learning rate parameter must be in (0,1]");
            if ( discount_ <= 0 || discount_ > 1 )  throw std::runtime_error("Discount parameter must be in (0,1]");
        }

        void QLearning::stepUpdateQ(size_t s, size_t s1, size_t a, double rew, QFunction * pq) {
            assert(pq != nullptr);

            QFunction & q = *pq;

            q[s][a] += alpha_ * ( rew + discount_ * (*std::max_element(std::begin(q[s1]),std::end(q[s1]))) - q[s][a] );
        }

        void QLearning::setLearningRate(double a) {
            if ( a <= 0.0 || a > 1.0 ) return;
            alpha_ = a;
        }

        double QLearning::getLearningRate() const {
            return alpha_;
        }

        void QLearning::setDiscount(double d) {
            if ( d <= 0.0 || d > 1.0 ) return;
            discount_ = d;
        }

        double QLearning::getDiscount() const {
            return discount_;
        }
    }
}
