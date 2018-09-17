#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>

namespace AIToolbox::POMDP {
    FastInformedBound::FastInformedBound(const unsigned horizon, const double tolerance) :
            horizon_(horizon)
    {
        setTolerance(tolerance);
    }

    void FastInformedBound::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    void FastInformedBound::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    double FastInformedBound::getTolerance()   const { return tolerance_; }
    unsigned FastInformedBound::getHorizon() const { return horizon_; }
}
