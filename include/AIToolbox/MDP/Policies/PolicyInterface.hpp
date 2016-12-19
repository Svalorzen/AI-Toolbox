#ifndef AI_TOOLBOX_MDP_POLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief Simple typedef for most of MDP's policy needs.
         */
        using PolicyInterface = ::AIToolbox::PolicyInterface<size_t, size_t, size_t>;
    }
}

#endif

