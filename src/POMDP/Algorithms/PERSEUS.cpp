#include <AIToolbox/POMDP/Algorithms/PERSEUS.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::POMDP {
    PERSEUS::PERSEUS(const size_t nBeliefs, const unsigned h, const double e) :
            beliefSize_(nBeliefs), horizon_(h),
            rand_(Impl::Seeder::getSeed())
    {
        setEpsilon(e);
    }

    void PERSEUS::setEpsilon(const double e) {
        if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
        epsilon_ = e;
    }

    void PERSEUS::setHorizon(const unsigned h) {
        horizon_ = h;
    }

    void PERSEUS::setBeliefSize(const size_t nBeliefs) {
        beliefSize_ = nBeliefs;
    }

    double PERSEUS::getEpsilon() const { return epsilon_; }
    unsigned PERSEUS::getHorizon() const { return horizon_; }
    size_t PERSEUS::getBeliefSize() const { return beliefSize_; }
}
