#include <AIToolbox/MDP/Algorithms/SARSAL.hpp>

namespace AIToolbox::MDP {
    SARSAL::SARSAL(const size_t ss, const size_t aa, const double discount, const double alpha, const double lambda, const double tolerance) :
            S(ss), A(aa), q_(makeQFunction(S, A))
    {
        setDiscount(discount);
        setLearningRate(alpha);
        setLambda(lambda);
        setTolerance(tolerance);
    }

    void SARSAL::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const size_t a1, const double rew) {
        const auto error = alpha_ * ( rew + discount_ * q_(s1, a1) - q_(s, a) );
        bool newTrace = true;

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
        for (size_t i = 0; i < traces_.size(); ++i) {
            auto & [ss, aa, el] = traces_[i];
            if (ss == s && aa == a) {
                el = 1.0;
                newTrace = false;
            } else {
                el *= gammaL_;
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

    void SARSAL::clearTraces() {
        traces_.clear();
    }

    const SARSAL::Traces & SARSAL::getTraces() const {
        return traces_;
    }

    void SARSAL::setTraces(const Traces & t) {
        traces_ = t;
    }

    void SARSAL::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double SARSAL::getLearningRate() const { return alpha_; }

    void SARSAL::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
        gammaL_ = lambda_ * discount_;
    }

    double SARSAL::getDiscount() const { return discount_; }

    void SARSAL::setLambda(const double lambda) {
        if ( lambda < 0.0 || lambda > 1.0 ) throw std::invalid_argument("Lambda parameter must be in [0,1]");

        lambda_ = lambda;
        gammaL_ = lambda_ * discount_;
    }

    double SARSAL::getLambda() const {
        return lambda_;
    }

    void SARSAL::setTolerance(const double tolerance) {
        tolerance_ = tolerance;
    }

    double SARSAL::getTolerance() const {
        return tolerance_;
    }

    size_t SARSAL::getS() const { return S; }
    size_t SARSAL::getA() const { return A; }

    const QFunction & SARSAL::getQFunction() const { return q_; }
    void SARSAL::setQFunction(const QFunction & qfun) { 
        assert(q_.rows() == qfun.rows());
        assert(q_.cols() == qfun.cols());
        q_ = qfun; 
    }
}
