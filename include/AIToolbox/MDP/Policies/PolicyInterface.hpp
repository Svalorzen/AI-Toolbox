#ifndef AI_TOOLBOX_MDP_POLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/PolicyInterface.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief Simple typedef for most of MDP's policy needs.
     */
    class PolicyInterface : public virtual AIToolbox::PolicyInterface<size_t, size_t, size_t> {
        public:
            using Base = AIToolbox::PolicyInterface<size_t, size_t, size_t>;

            /**
             * @brief This function returns a matrix containing all probabilities of the policy.
             *
             * Note that this may be expensive to compute, and should not
             * be called often (aside from the fact that it needs to
             * allocate a new Matrix2D each time).
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Matrix2D getPolicy() const = 0;
    };
}

#endif

