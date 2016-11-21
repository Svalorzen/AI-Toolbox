#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox {
    namespace POMDP {

        PBVI::PBVI(const size_t nBeliefs, const unsigned h, const double e) :
                beliefSize_(nBeliefs), horizon_(h), rand_(Impl::Seeder::getSeed())
        {
            setEpsilon(e);
        }

        void PBVI::setEpsilon(const double e) {
            if ( e < 0.0 ) throw std::invalid_argument("Epsilon must be >= 0");
            epsilon_ = e;
        }

        void PBVI::setHorizon(const unsigned h) {
            horizon_ = h;
        }

        void PBVI::setBeliefSize(const size_t nBeliefs) {
            beliefSize_ = nBeliefs;
        }

        double PBVI::getEpsilon() const { return epsilon_; }
        unsigned PBVI::getHorizon() const { return horizon_; }
        size_t PBVI::getBeliefSize() const { return beliefSize_; }
    }
}
