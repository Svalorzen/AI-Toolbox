#include <AIToolbox/POMDP/Algorithms/AMDP.hpp>

namespace AIToolbox::POMDP {
    AMDP::AMDP(const size_t nBeliefs, const size_t entropyBuckets) :
            beliefSize_(nBeliefs), buckets_(entropyBuckets) {}

    AMDP::Discretizer AMDP::makeDiscretizer(const size_t S) {
        // This is because lambdas are stupid and can't
        // capture member variables..
        const auto buckets = buckets_ - 1;
        return [S, buckets](const Belief & b) {
            // This stepsize is bounded by the minimum value entropy can take for a belief:
            // when the belief is uniform it would be: S * 1/S * log(1/S) = log(1/S)
            static const double stepSize = std::log(1.0/S) / static_cast<double>(buckets + 1);
            size_t maxS = 0;
            double entropy = 0.0;
            for ( size_t s = 0; s < S; ++s ) {
                if ( checkDifferentSmall(0.0, b[s]) ) {
                    entropy += b[s] * std::log(b[s]);
                    if ( b[s] > b[maxS] ) maxS = s;
                }
            }
            maxS += S * std::min(static_cast<size_t>(entropy / stepSize), buckets);
            return maxS;
        };
    }

    void AMDP::setBeliefSize(const size_t nBeliefs) {
        beliefSize_ = nBeliefs;
    }

    void AMDP::setEntropyBuckets(const size_t buckets) {
        buckets_ = buckets;
    }

    size_t AMDP::getBeliefSize() const {
        return beliefSize_;
    }

    size_t AMDP::getEntropyBuckets() const {
        return buckets_;
    }
}
