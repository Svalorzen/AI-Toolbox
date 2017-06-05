#include <AIToolbox/POMDP/Algorithms/QMDP.hpp>

namespace AIToolbox::POMDP {
    QMDP::QMDP(const unsigned horizon, const double epsilon) :
            solver_(horizon, epsilon) {}

    void QMDP::setEpsilon(const double e) {
        solver_.setEpsilon(e);
    }

    void QMDP::setHorizon(const unsigned h) {
        solver_.setHorizon(h);
    }

    double QMDP::getEpsilon() const {
        return solver_.getEpsilon();
    }

    unsigned QMDP::getHorizon() const {
        return solver_.getHorizon();
    }
}
