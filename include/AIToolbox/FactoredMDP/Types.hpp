#ifndef AI_TOOLBOX_FACTORED_MDP_TYPES_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>

#include <vector>
#include <utility>

namespace AIToolbox {
    namespace FactoredMDP {
        /**
         * @name Factored MDP Value Types
         *
         * @{
         */

        using State = std::vector<size_t>;
        using PartialState = std::vector<std::pair<size_t, size_t>>;

        // @}
    }
}

#endif
