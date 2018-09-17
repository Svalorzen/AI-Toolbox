#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::POMDP {
    PBVI::PBVI(const size_t nBeliefs, const unsigned h, const double t) :
            beliefSize_(nBeliefs), horizon_(h), rand_(Impl::Seeder::getSeed())
    {
        setTolerance(t);
    }

    void PBVI::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    void PBVI::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    void PBVI::setBeliefSize(const size_t nBeliefs) {
        beliefSize_ = nBeliefs;
    }

    double PBVI::getTolerance() const { return tolerance_; }
    unsigned PBVI::getHorizon() const { return horizon_; }
    size_t PBVI::getBeliefSize() const { return beliefSize_; }
}
