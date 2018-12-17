#include <AIToolbox/Bandit/Policies/QSoftmaxPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Bandit/Policies/Utils/QGreedyPolicyWrapper.hpp>

namespace AIToolbox::Bandit {
    QSoftmaxPolicy::QSoftmaxPolicy(const QFunction & q, const double t) :
            PolicyInterface::Base(q.size()),
            temperature_(t), q_(q)
    {
        if ( temperature_ < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
    }

    size_t QSoftmaxPolicy::sampleAction() const {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, bestActions_, rand_);
            return wrap.sampleAction();
        }

        Vector actionValues = (q_ / temperature_).array().exp();

        unsigned infinities = 0;
        for ( size_t a = 0; a < A; ++a )
            if ( std::isinf(actionValues(a)) )
                bestActions_[infinities++] = a;

        if (infinities) {
            auto pickDistribution = std::uniform_int_distribution<unsigned>(0, infinities-1);
            unsigned selection = pickDistribution(rand_);

            return bestActions_[selection];
        } else {
            actionValues /= actionValues.sum();

            return sampleProbability(A, actionValues, rand_);
        }
    }

    double QSoftmaxPolicy::getActionProbability(const size_t & a) const {
        if ( checkEqualSmall(temperature_, 0.0) ) {
            auto wrap = QGreedyPolicyWrapper(q_, bestActions_, rand_);
            return wrap.getActionProbability(a);
        }

        Vector actionValues = (q_ / temperature_).array().exp();

        unsigned infinities = 0;
        bool isAInfinite = false;
        double sum = 0.0;
        for ( size_t aa = 0; aa < A; ++aa ) {
            sum += actionValues(aa);
            if ( std::isinf(actionValues(aa)) ) {
                infinities++;
                isAInfinite |= (aa == a);
            }
        }

        if ( infinities ) {
            if ( isAInfinite ) return 1.0 / infinities;
            return 0.0;
        }

        return actionValues(a) / sum;
    }
}
