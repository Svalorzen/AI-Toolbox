#include <AIToolbox/MDP/Policies/QSoftmaxPolicy.hpp>

#include <cmath>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox {
    namespace MDP {
        QSoftmaxPolicy::QSoftmaxPolicy(const QFunction & q, const double t) :
                PolicyInterface::Base(q.rows(), q.cols()), QPolicyInterface(q),
                temperature_(t), greedy_(q)
        {
            if ( temperature_ < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
        }

        size_t QSoftmaxPolicy::sampleAction(const size_t & s) const {
            if ( checkEqualSmall(temperature_, 0.0) )
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
                actionValues /= actionValues.sum();

                return sampleProbability(A, actionValues, rand_);
            }
        }

        double QSoftmaxPolicy::getActionProbability(const size_t & s, const size_t & a) const {
            if ( checkEqualSmall(temperature_, 0.0) )
                return greedy_.getActionProbability(s, a);

            Vector actionValues(A);

            unsigned infinities = 0, isAInfinite = 0;
            double sum = 0.0;
            for ( size_t aa = 0; aa < A; ++aa ) {
                actionValues(aa) = std::exp(q_(s, aa) / temperature_);
                sum += actionValues(aa);
                if ( std::isinf(actionValues(aa)) ) {
                    infinities++;
                    if (aa == a)
                        isAInfinite = 1;
                }
            }

            if ( infinities ) {
                if ( isAInfinite ) return 1.0 / infinities;
                return 0.0;
            }

            if ( checkEqualSmall(sum, 0.0) )
                return 1.0 / A;

            return actionValues(a) / sum;
        }

        Matrix2D QSoftmaxPolicy::getPolicy() const {
            if ( checkEqualSmall(temperature_, 0.0) )
                return greedy_.getPolicy();

            Matrix2D retval(S, A);

            for (size_t s = 0; s < S; ++s) {
                unsigned infinities = 0;
                double sum = 0.0;
                for ( size_t a = 0; a < A; ++a ) {
                    retval(s, a) = std::exp(q_(s, a) / temperature_);
                    sum += retval(s, a);
                    if ( std::isinf(retval(s, a)) ) {
                        infinities++;
                    }
                }

                if ( infinities ) {
                    for ( size_t a = 0; a < A; ++a ) {
                        if ( std::isinf(retval(s, a)) )
                            retval(s, a) = 1.0 / infinities;
                        else
                            retval(s, a) = 0.0;
                    }
                    continue;
                }

                if ( checkEqualSmall(sum, 0.0) ) {
                    retval.row(s).fill(1.0 / A);
                    continue;
                }

                retval.row(s) /= sum;
            }
            return retval;
        }

        void QSoftmaxPolicy::setTemperature(const double t) {
            if ( t < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
            temperature_ = t;
        }

        double QSoftmaxPolicy::getTemperature() const {
            return temperature_;
        }
    }
}
