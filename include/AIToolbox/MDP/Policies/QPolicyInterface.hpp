#ifndef AI_TOOLBOX_MDP_QPOLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_MDP_QPOLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
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
        class QPolicyInterface : public PolicyInterface<size_t> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param q The QFunction this policy is linked with.
                 */
                QPolicyInterface(const QFunction & q);

                /**
                 * @brief This function chooses an action for state s, following the policy distribution.
                 *
                 * @param s The sampled state of the policy.
                 *
                 * @return The chosen action.
                 */
                virtual size_t sampleAction(const size_t & s) const override = 0;

                /**
                 * @brief This function returns the probability of taking the specified action in the specified state.
                 *
                 * @param s The selected state.
                 * @param a The selected action.
                 *
                 * @return The probability of taking the selected action in the specified state.
                 */
                virtual double getActionProbability(const size_t & s, size_t a) const override = 0;

            protected:
                const QFunction & q_;
        };
    }
}

#endif
