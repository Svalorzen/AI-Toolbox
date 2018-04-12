#ifndef AI_TOOLBOX_FACTORED_MDP_SINGLE_ACTION_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_SINGLE_ACTION_POLICY_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>
#include <AIToolbox/Factored/MDP/Types.hpp>

namespace AIToolbox::Factored::MDP {
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
    class SingleActionPolicy : public PolicyInterface<State, State, Action> {
        public:
            using Base = PolicyInterface<State, State, Action>;

            /**
             * @brief Basic constructor.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            SingleActionPolicy(State s, Action a);

            /**
             * @brief This function always return the current action.
             *
             * @param s The unused sampled state of the policy.
             *
             * @return The currently saved action.
             */
            virtual Action sampleAction(const State & s) const;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return One if the action matches the currently saved one, zero otherwise.
             */
            virtual double getActionProbability(const State & s, const Action & a) const;

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
