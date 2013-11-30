#ifndef AI_TOOLBOX_MDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_MDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    class Policy;
    namespace MDP {
        /**
         * @brief This function creates a policy from a given QFunction.
         *
         * @param q The QFunction that is begin read.
         *
         * @return A new Policy.
         */
        Policy makePolicy(const QFunction & q);
    }
}

#endif
