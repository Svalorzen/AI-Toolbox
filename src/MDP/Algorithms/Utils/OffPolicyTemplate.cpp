#include <AIToolbox/MDP/Algorithms/Utils/OffPolicyTemplate.hpp>

#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    OffPolicyBase::OffPolicyBase(const size_t s, const size_t a, const double discount, const double alpha, const double tolerance) :
            S(s), A(a), q_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setLearningRate(alpha);
        setTolerance(tolerance);
    }

    void OffPolicyBase::updateTraces(const size_t s, const size_t a, const double error, const double traceDiscount) {
        // So basically here we have in traces_ a non-ordered list of the old
        // state/action pairs we have already seen. For each item in this list,
        // we scale its "relevantness" back by gammaL_, and we update its
        // q-value accordingly.
        //
        // If the current s-a are in the list already, their eligibility is
        // directly updated to 1.0. Otherwise, they are added to the list.
        //
        // If any element would become too far away temporally to still be
        // relevant, we extract it from the list. As the order is not important
        // (it is implicit in the "el" element), we can use swap+pop.
        bool newTrace = true;
        for (size_t i = 0; i < traces_.size(); ++i) {
            auto & [ss, aa, el] = traces_[i];
            if (ss == s && aa == a) {
                el = 1.0;
                newTrace = false;
            } else {
                el *= traceDiscount;
                if (el < tolerance_) {
                    std::swap(traces_[i], traces_[traces_.size() - 1]);
                    traces_.pop_back();
                    --i;
                    continue;
                }
            }
            q_(ss, aa) += error * el;
        }
        if (newTrace) {
            traces_.emplace_back(s, a, 1.0);
            q_(s, a) += error; // el is 1.0 here
        }
    }

    void OffPolicyBase::clearTraces() {
        traces_.clear();
    }

    const OffPolicyBase::Traces & OffPolicyBase::getTraces() const {
        return traces_;
    }

    void OffPolicyBase::setTraces(const Traces & t) {
        traces_ = t;
    }

    void OffPolicyBase::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double OffPolicyBase::getLearningRate() const { return alpha_; }

    void OffPolicyBase::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    double OffPolicyBase::getDiscount() const { return discount_; }

    void OffPolicyBase::setTolerance(const double t) {
        tolerance_ = t;
    }

    double OffPolicyBase::getTolerance() const {
        return tolerance_;
    }

    size_t OffPolicyBase::getS() const { return S; }
    size_t OffPolicyBase::getA() const { return A; }

    const QFunction & OffPolicyBase::getQFunction() const { return q_; }
    void OffPolicyBase::setQFunction(const QFunction & qfun) { q_ = qfun; }
}
