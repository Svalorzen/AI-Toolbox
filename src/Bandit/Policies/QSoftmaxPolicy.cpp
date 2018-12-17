#include <AIToolbox/Bandit/Policies/QSoftmaxPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Bandit/Policies/Utils/QSoftmaxPolicyWrapper.hpp>

namespace AIToolbox::Bandit {
    QSoftmaxPolicy::QSoftmaxPolicy(const QFunction & q, const double t) :
            PolicyInterface::Base(q.size()),
            temperature_(t), q_(q), bestActions_(A), vbuffer_(A)
    {
        if ( temperature_ < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
    }

    size_t QSoftmaxPolicy::sampleAction() const {
        auto wrap = QSoftmaxPolicyWrapper(temperature_, q_, vbuffer_, bestActions_, rand_);
        return wrap.sampleAction();
    }

    double QSoftmaxPolicy::getActionProbability(const size_t & a) const {
        auto wrap = QSoftmaxPolicyWrapper(temperature_, q_, vbuffer_, bestActions_, rand_);
        return wrap.getActionProbability(a);
    }

    Vector QSoftmaxPolicy::getPolicy() const {
        auto wrap = QSoftmaxPolicyWrapper(temperature_, q_, vbuffer_, bestActions_, rand_);

        Vector retval(A);
        wrap.getPolicy(retval);

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
