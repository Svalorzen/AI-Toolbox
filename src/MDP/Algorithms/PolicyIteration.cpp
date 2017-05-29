#include <AIToolbox/MDP/Algorithms/PolicyIteration.hpp>

namespace AIToolbox::MDP {
    PolicyIteration::PolicyIteration(unsigned horizon, double epsilon) :
            horizon_(horizon)
    {
        setEpsilon(epsilon);
    }

    void PolicyIteration::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    void PolicyIteration::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    double PolicyIteration::getEpsilon()   const { return epsilon_; }

    unsigned PolicyIteration::getHorizon() const { return horizon_; }
}
