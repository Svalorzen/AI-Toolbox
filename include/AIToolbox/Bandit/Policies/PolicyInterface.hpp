#ifndef AI_TOOLBOX_BANDIT_POLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_BANDIT_POLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief Simple typedef for most of a normal Bandit's policy needs.
     */
    class PolicyInterface : public virtual AIToolbox::PolicyInterface<void, void, size_t> {
        public:
            using Base = AIToolbox::PolicyInterface<void, void, size_t>;
    };
}

#endif
