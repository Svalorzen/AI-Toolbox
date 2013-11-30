#ifndef AI_TOOLBOX_MDP_QPOLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_QPOLICY_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    class Policy;
    namespace MDP {
        /**
         * @brief This class models a policy through a QFunction.
         * 
         * This function provides a way to sample actions without the
         * need to compute a full Policy from a QFunction. This is useful
         * because often many methods need to modify small parts of a Qfunction
         * for progressive improvement, and computing a full Policy at each
         * step can become too expensive to do.
         */
        class QPolicy : public PolicyInterface {
            public:

                /**
                 * @brief Basic constructor.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param q The QFunction this policy is linked with.
                 */
                QPolicy(const QFunction & q);

                /**
                 * @brief This function chooses a random action for state s, following the policy distribution.
                 *
                 * @param s The sampled state of the policy.
                 *
                 * @return The chosen action.
                 */
                virtual size_t sampleAction(size_t s) const;

                /**
                 * @brief This function returns the probability of taking the specified action in the specified state.
                 *
                 * @param s The selected state.
                 * @param a The selected action.
                 *
                 * @return The probability of taking the selected action in the specified state.
                 */
                virtual double getActionProbability(size_t s, size_t a) const;

            private:
                const QFunction & q_;
        };
    }
}

#endif
