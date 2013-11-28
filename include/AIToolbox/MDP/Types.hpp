#ifndef AI_TOOLBOX_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_MDP_TYPES_HEADER_FILE

#include <vector>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        using ValueFunction     = std::vector<double>;
        using QFunction         = Table2D;
    }
}

#endif
