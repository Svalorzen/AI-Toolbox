#include <AIToolbox/MDP/QLearning.hpp>

#include <cassert>

namespace AIToolbox {
    namespace MDP {
         
        QLearning::QLearning(double discount) : discount_(discount) {
            if ( discount_ <= 0 || discount_ > 1 )  throw std::runtime_error("Discount parameter must be in (0,1]");
        }

        void QLearning::stepUpdateQ(size_t s, size_t s1, size_t a, double rew, QFunction * pq) {
            assert(pq != nullptr);

            QFunction & q = *pq;

            q[s][a] += discount_ * ( rew * (*std::max_element(std::begin(q[s1]),std::end(q[s1]))) - q[s][a] );
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
