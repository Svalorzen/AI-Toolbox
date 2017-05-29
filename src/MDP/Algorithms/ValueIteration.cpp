#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

namespace AIToolbox::MDP {
    ValueIteration::ValueIteration(unsigned horizon, double epsilon, ValueFunction v) :
            horizon_(horizon), vParameter_(v)
    {
        setEpsilon(epsilon);
    }

    void ValueIteration::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    void ValueIteration::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    void ValueIteration::setValueFunction(ValueFunction v) {
        vParameter_ = std::move(v);
    }

    double ValueIteration::getEpsilon()   const { return epsilon_; }

    unsigned ValueIteration::getHorizon() const { return horizon_; }

    const ValueFunction & ValueIteration::getValueFunction() const { return vParameter_; }
}
