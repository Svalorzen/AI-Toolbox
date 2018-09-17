#include <AIToolbox/POMDP/Algorithms/PERSEUS.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::POMDP {
    PERSEUS::PERSEUS(const size_t nBeliefs, const unsigned h, const double t) :
            beliefSize_(nBeliefs), horizon_(h),
            rand_(Impl::Seeder::getSeed())
    {
        setTolerance(t);
    }

    void PERSEUS::setTolerance(const double t) {
        if ( t < 0.0 ) throw std::invalid_argument("Tolerance must be >= 0");
        tolerance_ = t;
    }

    void PERSEUS::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    void PERSEUS::setBeliefSize(const size_t nBeliefs) {
        beliefSize_ = nBeliefs;
    }

    double PERSEUS::getTolerance() const { return tolerance_; }
    unsigned PERSEUS::getHorizon() const { return horizon_; }
    size_t PERSEUS::getBeliefSize() const { return beliefSize_; }
}
