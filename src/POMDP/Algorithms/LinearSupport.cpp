#include <AIToolbox/POMDP/Algorithms/LinearSupport.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::POMDP {
    bool LinearSupport::VertexComparator::operator()(const Vertex & lhs, const Vertex & rhs) const {
        return lhs.error < rhs.error;
    }

    // -----

    LinearSupport::LinearSupport(const unsigned h, const double t) : horizon_(h) {
        setTolerance(t);
    }

    void LinearSupport::setHorizon(const unsigned h) {
        horizon_ = h;
    }
    void LinearSupport::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    unsigned LinearSupport::getHorizon() const {
        return horizon_;
    }

    double LinearSupport::getTolerance() const {
        return tolerance_;
    }
}
