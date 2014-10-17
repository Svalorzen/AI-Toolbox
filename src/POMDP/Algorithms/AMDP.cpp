#include <AIToolbox/POMDP/Algorithms/AMDP.hpp>

namespace AIToolbox {
    namespace POMDP {
        AMDP::AMDP(size_t nBeliefs, size_t entropyBuckets) : beliefSize_(nBeliefs), buckets_(entropyBuckets) {}

        void AMDP::setBeliefSize(size_t nBeliefs) {
            beliefSize_ = nBeliefs;
        }

        void AMDP::setEntropyBuckets(size_t buckets) {
            buckets_ = buckets;
        }

        size_t AMDP::getBeliefSize() const {
            return beliefSize_;
        }

        size_t AMDP::getEntropyBuckets() const {
            return buckets_;
        }
    }
}
