#ifndef AI_TOOLBOX_BANDIT_RANDOM_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_RANDOM_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class represents a random policy.
     *
     * This class simply returns a random action every time it is polled.
     */
    class RandomPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param a The number of actions available to the agent.
             */
            RandomPolicy(size_t a);

            /**
             * @brief This function chooses a random action for state s, following the policy distribution.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const size_t & a) const override;

            /**
             * @brief This function returns a vector containing all probabilities of the policy.
             *
             * Note that this may be expensive to compute, and should not
             * be called often (aside from the fact that it needs to
             * allocate a new Vector each time).
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Vector getPolicy() const override;

        private:
            // Used to sampled random actions
            mutable std::uniform_int_distribution<size_t> randomDistribution_;
    };
}

#endif
