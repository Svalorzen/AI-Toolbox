#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    namespace MDP {
        Policy makePolicy(size_t S, size_t A, const QFunction & q) {
            Policy p(S,A);
            for ( size_t s = 0; s < S; s++ ) {
                auto it = std::max_element(std::begin(q[s]), std::end(q[s]));
                p.setPolicy(s, std::distance(std::begin(q[s]), it));
            }
            return p;
        }
    }
}
