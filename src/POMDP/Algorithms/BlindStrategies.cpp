#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>

namespace AIToolbox::POMDP {
    BlindStrategies::BlindStrategies(const unsigned horizon, const double tolerance) :
            horizon_(horizon)
    {
        setTolerance(tolerance);
    }

    void BlindStrategies::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    void BlindStrategies::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    double BlindStrategies::getTolerance()   const { return tolerance_; }
    unsigned BlindStrategies::getHorizon() const { return horizon_; }
}
