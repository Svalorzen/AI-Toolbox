#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox {
    namespace POMDP {

        PBVI::PBVI(size_t nBeliefs, unsigned h) : beliefSize_(nBeliefs), horizon_(h), rand_(Impl::Seeder::getSeed()) {}

        void PBVI::setHorizon(unsigned h) {
            horizon_ = h;
        }

        void PBVI::setBeliefSize(size_t nBeliefs) {
            beliefSize_ = nBeliefs;
        }

        unsigned PBVI::getHorizon() const { return horizon_; }
        size_t PBVI::getBeliefSize() const { return beliefSize_; }
    }
}
