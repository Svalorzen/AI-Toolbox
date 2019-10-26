#ifndef AI_TOOLBOX_MDP_RANDOM_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_RANDOM_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Policies/RandomPolicy.hpp>
#include <AIToolbox/MDP/Policies/BanditPolicyAdaptor.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents a random policy.
     *
     * This class simply returns a random action every time it is polled.
     */
    using RandomPolicy = BanditPolicyAdaptor<Bandit::RandomPolicy>;
}

#endif
