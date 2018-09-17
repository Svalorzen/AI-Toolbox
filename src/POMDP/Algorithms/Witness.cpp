#include <AIToolbox/POMDP/Algorithms/Witness.hpp>

namespace AIToolbox::POMDP {
    Witness::Witness(const unsigned h, const double t) : horizon_(h) {
        setTolerance(t);
    }

    void Witness::setHorizon(const unsigned h) {
        horizon_ = h;
    }
    void Witness::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    unsigned Witness::getHorizon() const {
        return horizon_;
    }

    double Witness::getTolerance() const {
        return tolerance_;
    }
}
