#include <AIToolbox/POMDP/Algorithms/QMDP.hpp>

namespace AIToolbox {
    namespace POMDP {
        QMDP::QMDP(unsigned horizon, double epsilon) : solver_(horizon, epsilon) {}

        void QMDP::setEpsilon(double e) {
            solver_.setEpsilon(e);
        }

        void QMDP::setHorizon(unsigned h) {
            solver_.setHorizon(h);
        }

        double QMDP::getEpsilon() const {
            return solver_.getEpsilon();
        }

        unsigned QMDP::getHorizon() const {
            return solver_.getHorizon();
        }
    }
}
