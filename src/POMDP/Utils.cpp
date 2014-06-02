#include <AIToolbox/POMDP/Utils.hpp>

namespace AIToolbox {
    namespace POMDP {

        VEntry makeVEntry(size_t S, size_t a, size_t O) {
            return std::make_tuple(MDP::Values(S, 0.0), a, VObs(O, 0));
        }

    }
}
