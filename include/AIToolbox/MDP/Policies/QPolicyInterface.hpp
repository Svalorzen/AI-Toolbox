#ifndef AI_TOOLBOX_MDP_QPOLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_QPOLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class is an interface to specify a policy through a QFunction.
     *
     * This class provides a way to sample actions without the
     * need to compute a full Policy from a QFunction. This is useful
     * because often many methods need to modify small parts of a Qfunction
     * for progressive improvement, and computing a full Policy at each
     * step can become too expensive to do.
     *
     * The type of policy obtained from such sampling is left to the implementation,
     * since there are many ways in which such a policy may be formed.
     */
    class QPolicyInterface : public virtual PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param q The QFunction this policy is linked with.
             */
            QPolicyInterface(const QFunction & q);

            /**
             * @brief This function returns the underlying QFunction reference.
             *
             * @return The underlying QFunction reference.
             */
            const QFunction & getQFunction() const;

        protected:
            const QFunction & q_;
    };
}

#endif
