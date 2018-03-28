#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>

namespace AIToolbox::POMDP {
    FastInformedBound::FastInformedBound(const unsigned horizon, const double epsilon) :
            horizon_(horizon)
    {
        setEpsilon(epsilon);
    }

    void FastInformedBound::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    void FastInformedBound::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    double FastInformedBound::getEpsilon()   const { return epsilon_; }
    unsigned FastInformedBound::getHorizon() const { return horizon_; }
}
