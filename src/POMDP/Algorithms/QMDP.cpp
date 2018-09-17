#include <AIToolbox/POMDP/Algorithms/QMDP.hpp>

namespace AIToolbox::POMDP {
    QMDP::QMDP(const unsigned horizon, const double tolerance) :
            solver_(horizon, tolerance) {}

    void QMDP::setTolerance(const double t) {
        solver_.setTolerance(t);
    }

    void QMDP::setHorizon(const unsigned h) {
        solver_.setHorizon(h);
    }

    double QMDP::getTolerance() const {
        return solver_.getTolerance();
    }

    unsigned QMDP::getHorizon() const {
        return solver_.getHorizon();
    }

    VList QMDP::fromQFunction(const size_t O, const MDP::QFunction & qfun) {
        const auto A = static_cast<size_t>(qfun.cols());

        VList w;
        w.reserve(A);

        // We simply create a VList where each entry is a slice of the
        // MDP::QFunction, one per action. In the QFunction, each column
        // contains the values of an action in all states. In a POMDP, one can
        // see this as a plane connecting the corners of the simplex at those
        // particular values. The values in the middle are deduced from that
        // linearly.
        for ( size_t a = 0; a < A; ++a ) {
            // All observations are 0 since we go back to the horizon 0 entry,
            // which is nil.
            // Here we only make the VList, but we do it so that if it's later
            // embedded into a ValueFunction it'll magically work without
            // crashing.
            w.emplace_back(qfun.col(a), a, VObs(O, 0u));
        }

        return w;
    }
}
