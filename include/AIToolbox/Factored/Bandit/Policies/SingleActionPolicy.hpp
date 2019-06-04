#ifndef AI_TOOLBOX_FACTORED_BANDIT_SINGLE_ACTION_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_SINGLE_ACTION_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents a policy always picking the same action.
     *
     * Since there are methods which in order to learn automatically
     * compute a best action for the next time step, it is useful to be
     * able to wrap those actions into a policy in order to be joined to
     * other policies (like epsilon-greedy, for example).
     *
     * This class is a simple wrapper that always return the last action
     * that has been set.
     */
    class SingleActionPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param a The number of actions available to the agent.
             */
            SingleActionPolicy(Action a);

            /**
             * @brief This function always return the current action.
             *
             * @return The currently saved action.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param a The selected action.
             *
             * @return One if the action matches the currently saved one, zero otherwise.
             */
            virtual double getActionProbability(const Action & a) const override;

            /**
             * @brief This function updates the currently hold action.
             *
             * @param a The new action we must return.
             */
            void updateAction(Action a);

        private:
            // The only action returned by this policy.
            Action currentAction_;
    };
}

#endif
