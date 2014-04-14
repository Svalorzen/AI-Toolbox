#ifndef AI_TOOLBOX_MDP_EPSILON_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_EPSILON_POLICY_HEADER_FILE

#include <AIToolbox/EpsilonPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        using EpsilonPolicy = ::AIToolbox::EpsilonPolicyInterface<size_t>;
    }
}

#endif
