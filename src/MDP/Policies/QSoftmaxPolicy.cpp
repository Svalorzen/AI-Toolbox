#include <AIToolbox/MDP/Policies/QSoftmaxPolicy.hpp>

#include <cmath>

#include <AIToolbox/Utils.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        QSoftmaxPolicy::QSoftmaxPolicy(const QFunction & q, double t) : QPolicyInterface(q), temperature_(t), greedy_(q) {
            if ( temperature_ < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
        }

        size_t QSoftmaxPolicy::sampleAction(const size_t & s) const {
            if ( temperature_ == 0 )
                return greedy_.sampleAction(s);

            Vector actionValues(A);

            for ( size_t a = 0; a < A; ++a )
                actionValues(a) = std::exp(q_(s, a) / temperature_);

            actionValues.normalize();

            return sampleProbability(A, actionValues, rand_);
        }

        double QSoftmaxPolicy::getActionProbability(const size_t & s, size_t a) const {
            if ( temperature_ == 0.0 )
                return greedy_.getActionProbability(s, a);

            Vector actionValues(A);

            for ( size_t aa = 0; aa < A; ++aa )
                actionValues(aa) = std::exp(q_(s, aa) / temperature_);

            return actionValues(a) / actionValues.sum();
        }

        void QSoftmaxPolicy::setTemperature(double t) {
            if ( t < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
            temperature_ = t;
        }

        double QSoftmaxPolicy::getTemperature() const {
            return temperature_;
        }
    }
}
