#ifndef AI_TOOLBOX_MDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_MDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        QFunction     makeQFunction    (size_t S, size_t A);
        ValueFunction makeValueFunction(size_t S);
    }
}

#endif
