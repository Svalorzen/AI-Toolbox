#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

namespace AIToolbox::MDP {
    ValueIteration::ValueIteration(unsigned horizon, double tolerance, ValueFunction v) :
            horizon_(horizon), vParameter_(v)
    {
        setTolerance(tolerance);
    }

    void ValueIteration::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    void ValueIteration::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    void ValueIteration::setValueFunction(ValueFunction v) {
        vParameter_ = std::move(v);
    }

    double ValueIteration::getTolerance()   const { return tolerance_; }

    unsigned ValueIteration::getHorizon() const { return horizon_; }

    const ValueFunction & ValueIteration::getValueFunction() const { return vParameter_; }
}
