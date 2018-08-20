#include <AIToolbox/POMDP/Algorithms/LinearSupport.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::POMDP {
    bool LinearSupport::Comparator::operator()(const Belief & lhs, const Belief & rhs) const {
        return veccmpSmall(lhs, rhs) < 0;
    }

    bool LinearSupport::Comparator::operator()(const Vertex & lhs, const Vertex & rhs) const {
        return lhs.error < rhs.error;
    }

    // -----

    LinearSupport::LinearSupport(const unsigned h, const double e) : horizon_(h) {
        setEpsilon(e);
    }

    void LinearSupport::setHorizon(const unsigned h) {
        horizon_ = h;
    }
    void LinearSupport::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    unsigned LinearSupport::getHorizon() const {
        return horizon_;
    }

    double LinearSupport::getEpsilon() const {
        return epsilon_;
    }
}
