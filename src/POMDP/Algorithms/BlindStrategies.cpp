#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>

namespace AIToolbox::POMDP {
    BlindStrategies::BlindStrategies(const unsigned horizon, const double epsilon) :
            horizon_(horizon)
    {
        setEpsilon(epsilon);
    }

    void BlindStrategies::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    void BlindStrategies::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    double BlindStrategies::getEpsilon()   const { return epsilon_; }
    unsigned BlindStrategies::getHorizon() const { return horizon_; }
}
