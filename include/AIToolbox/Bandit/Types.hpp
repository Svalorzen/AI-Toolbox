#ifndef AI_TOOLBOX_BANDIT_TYPES_HEADER_FILE
#define AI_TOOLBOX_BANDIT_TYPES_HEADER_FILE

#include <AIToolbox/Types.hpp>

namespace AIToolbox::Bandit {
    /**
     * @name MDP Value Types
     *
     * The QFunction here is simply the vector containing the mean reward for
     * each action.
     *
     * @{
     */

    using QFunction = Vector;

    /** @}  */
}

#endif
