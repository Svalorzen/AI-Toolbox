#include <AIToolbox/MDP/Policies/QSoftmaxPolicy.hpp>

#include <AIToolbox/Bandit/Policies/Utils/QSoftmaxPolicyWrapper.hpp>

namespace AIToolbox::MDP {
    QSoftmaxPolicy::QSoftmaxPolicy(const QFunction & q, const double t) :
            PolicyInterface::Base(q.rows(), q.cols()), QPolicyInterface(q),
            temperature_(t), bestActions_(A), vbuffer_(A)
    {
        if ( temperature_ < 0.0 ) throw std::invalid_argument("Temperature must be >= 0");
    }

    size_t QSoftmaxPolicy::sampleAction(const size_t & s) const {
        auto wrap = Bandit::QSoftmaxPolicyWrapper(temperature_, q_.row(s), vbuffer_, bestActions_, rand_);
        return wrap.sampleAction();
    }

    double QSoftmaxPolicy::getActionProbability(const size_t & s, const size_t & a) const {
        auto wrap = Bandit::QSoftmaxPolicyWrapper(temperature_, q_.row(s), vbuffer_, bestActions_, rand_);
        return wrap.getActionProbability(a);
    }

    Matrix2D QSoftmaxPolicy::getPolicy() const {
        Matrix2D retval(S, A);

        for (size_t s = 0; s < S; ++s) {
            auto wrap = Bandit::QSoftmaxPolicyWrapper(temperature_, q_.row(s), vbuffer_, bestActions_, rand_);
            wrap.getPolicy(retval.row(s));
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
