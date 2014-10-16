#include <AIToolbox/POMDP/Algorithms/AMDP.hpp>

namespace AIToolbox {
    namespace POMDP {
        AMDP::AMDP(size_t nBeliefs, size_t entropyBuckets) : beliefSize_(nBeliefs), buckets_(entropyBuckets) {}
    }
}
