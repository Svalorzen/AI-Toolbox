#ifndef AI_TOOLBOX_FACTORED_BANDIT_POLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_POLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/Factored/Types.hpp>
#include <AIToolbox/PolicyInterface.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief Simple typedef for most of a normal Bandit's policy needs.
     */
    class PolicyInterface : public virtual AIToolbox::PolicyInterface<void, void, Action> {
        public:
            using Base = AIToolbox::PolicyInterface<void, void, Action>;
    };
}

#endif
