#include <AIToolbox/POMDP/Algorithms/PERSEUS.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox {
    namespace POMDP {

        PERSEUS::PERSEUS(size_t nBeliefs, unsigned h) : beliefSize_(nBeliefs), horizon_(h), rand_(Impl::Seeder::getSeed()) {}

        void PERSEUS::setHorizon(unsigned h) {
            horizon_ = h;
        }

        void PERSEUS::setBeliefSize(size_t nBeliefs) {
            beliefSize_ = nBeliefs;
        }

        unsigned PERSEUS::getHorizon() const { return horizon_; }
        size_t PERSEUS::getBeliefSize() const { return beliefSize_; }
    }
}
