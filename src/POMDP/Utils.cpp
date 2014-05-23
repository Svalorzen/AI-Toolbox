#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox {
    namespace POMDP {
        VList crossSum(size_t S, size_t a, const VList & l1, const VList & l2) {
            VList c;

            if ( ! ( l1.size() && l2.size() ) ) return c;

            for ( auto & v1 : l1 )
                for ( auto & v2 : l2 ) {
                    MDP::ValueFunction v(S, 0.0);
                    for ( size_t i = 0; i < S; ++i )
                        v[i] = v1.second[i] + v2.second[i];
                    c.emplace_back(a, std::move(v));
                }

             return c;
        }
    }
}
