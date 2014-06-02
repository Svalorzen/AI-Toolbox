#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        VEntry makeVEntry(size_t S, size_t a = 0, size_t O = 0);
    }
}

#endif
