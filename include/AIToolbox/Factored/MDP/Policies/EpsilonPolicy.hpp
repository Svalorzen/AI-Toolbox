#ifndef AI_TOOLBOX_FACTORED_MDP_EPSILON_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_EPSILON_POLICY_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/EpsilonPolicyInterface.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class represents an epsilon-greedy policy for Factored MDPs
     */
    class EpsilonPolicy : public EpsilonPolicyInterface<State, State, Action> {
        public:
            using EpsilonBase = EpsilonPolicyInterface<State, State, Action>;

            /**
             * @brief Basic constructor.
             *
             * This constructor saves the input policy and the epsilon
             * parameter for later use.
             *
             * The epsilon parameter must be >= 0.0 and <= 1.0,
             * otherwise the constructor will throw an std::invalid_argument.
             *
             * @param p The policy that is being extended.
             * @param epsilon The parameter that controls the amount of exploration.
             */
            EpsilonPolicy(const EpsilonBase::Base & p, double epsilon = 0.1);

        protected:
            /**
             * @brief This function returns a random action in the Action space.
             *
             * @return A valid random action.
             */
            virtual Action sampleRandomAction() const;

            /**
             * @brief This function returns the probability of picking a random action.
             *
             * @return The probability of picking an an action at random.
             */
            virtual double getRandomActionProbability() const;

            // Used to sampled random actions
            mutable std::vector<std::uniform_int_distribution<size_t>> randomDistribution_;
    };
}

#endif
