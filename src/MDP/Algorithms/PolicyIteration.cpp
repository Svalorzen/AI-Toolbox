#include <AIToolbox/MDP/Algorithms/PolicyIteration.hpp>

namespace AIToolbox::MDP {
    PolicyIteration::PolicyIteration(const unsigned horizon, const double tolerance) :
            horizon_(horizon)
    {
        setTolerance(tolerance);
    }

    void PolicyIteration::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    void PolicyIteration::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    double PolicyIteration::getTolerance()   const { return tolerance_; }

    unsigned PolicyIteration::getHorizon() const { return horizon_; }
}
