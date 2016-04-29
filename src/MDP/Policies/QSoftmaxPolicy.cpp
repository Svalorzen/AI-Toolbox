#include <AIToolbox/MDP/Policies/QSoftmaxPolicy.hpp>

#include <cmath>

#include <AIToolbox/Utils.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

#include <iostream>

namespace AIToolbox {
    namespace MDP {
        QSoftmaxPolicy::QSoftmaxPolicy(const QFunction & q, double t) : QPolicyInterface(q), temperature_(t), greedy_(q) {
            if ( temperature_ < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
        }

        size_t QSoftmaxPolicy::sampleAction(const size_t & s) const {
            if ( temperature_ == 0.0 )
                return greedy_.sampleAction(s);

            Vector actionValues(A);

            unsigned infinities = 0;
            for ( size_t a = 0; a < A; ++a ) {
                actionValues(a) = std::exp(q_(s, a) / temperature_);
                if ( std::isinf(actionValues(a)) )
                    infinities++;
            }

            if (infinities) {
                auto pickDistribution = std::uniform_int_distribution<unsigned>(0, infinities-1);
                unsigned selection = pickDistribution(rand_);

                size_t retval = 0;
                for ( ; retval < A - 1; ++retval) {
                    if ( std::isinf(actionValues(retval)) && !selection )
                        break;
                    --selection;
                }
                return retval;
            } else {
                actionValues.normalize();

                return sampleProbability(A, actionValues, rand_);
            }
        }

        double QSoftmaxPolicy::getActionProbability(const size_t & s, size_t a) const {
            if ( temperature_ == 0.0 )
                return greedy_.getActionProbability(s, a);

            Vector actionValues(A);

            unsigned infinities = 0, isAInfinite = 0;
            for ( size_t aa = 0; aa < A; ++aa ) {
                actionValues(aa) = std::exp(q_(s, aa) / temperature_);
                if ( std::isinf(actionValues(aa)) )
                    infinities++;
                if ( aa == a && std::isinf(actionValues(aa)) )
                    isAInfinite = 1;
            }

            if ( infinities ) {
                if ( isAInfinite ) return 1.0 / infinities;
                return 0.0;
            }

            auto sum = actionValues.sum();
            if ( checkEqualSmall(sum, 0.0) )
                return 1.0 / A;

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
