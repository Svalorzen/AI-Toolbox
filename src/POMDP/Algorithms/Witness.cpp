#include <AIToolbox/POMDP/Algorithms/Witness.hpp>

namespace AIToolbox::POMDP {
    Witness::Witness(const unsigned h, const double e) : horizon_(h) {
        setEpsilon(e);
    }

    void Witness::setHorizon(const unsigned h) {
        horizon_ = h;
    }
    void Witness::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    unsigned Witness::getHorizon() const {
        return horizon_;
    }

    double Witness::getEpsilon() const {
        return epsilon_;
    }
}
