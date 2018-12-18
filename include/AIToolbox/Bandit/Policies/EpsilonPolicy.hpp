#ifndef AI_TOOLBOX_BANDIT_EPSILON_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_EPSILON_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/EpsilonPolicyInterface.hpp>

namespace AIToolbox::Bandit {
    class EpsilonPolicy : public PolicyInterface, public EpsilonPolicyInterface<void, void, size_t> {
        public:
            using EpsilonBase = EpsilonPolicyInterface<void, void, size_t>;

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
            EpsilonPolicy(const PolicyInterface & p, double epsilon = 0.1);

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
            virtual Vector getPolicy() const override;

        protected:
            /**
             * @brief This function returns a random action in the Action space.
             *
             * @return A valid random action.
             */
            virtual size_t sampleRandomAction() const override;

            /**
             * @brief This function returns the probability of picking a random action.
             *
             * @return The probability of picking an an action at random.
             */
            virtual double getRandomActionProbability() const override;

            // Used to sampled random actions
            mutable std::uniform_int_distribution<size_t> randomDistribution_;
    };
}

#endif
