#ifndef AI_TOOLBOX_FACTORED_BANDIT_RANDOM_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_RANDOM_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Factored::Bandit {
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
            RandomPolicy(Action a);

            /**
             * @brief This function chooses a random action for state s, following the policy distribution.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const Action & a) const override;

        private:
            // Used to sampled random actions
            mutable std::vector<std::uniform_int_distribution<size_t>> randomDistributions_;
    };
}

#endif
