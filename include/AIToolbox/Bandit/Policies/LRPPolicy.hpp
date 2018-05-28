#ifndef AI_TOOLBOX_BANDIT_LRP_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_LRP_POLICY_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class models the Linear Reward Penalty algorithm.
     *
     * This algorithm performs direct policy updates depending on whether a
     * given action was a success or a penalty.
     *
     * In particular, the version called "Linear Reward-Inaction" (where the
     * 'b' parameter is set to zero) is guaranteed to converge to optimal in a
     * stationary environment.
     *
     * Additionally, this algorithm can also be used in multi-agent settings,
     * and will usually result in the convergence to some Nash equilibria.
     *
     * The successful updates are in the form:
     *
     *     p(t + 1) = p(t) + a * (1 − p(t))          // For the action taken
     *     p(t + 1) = p(t) − a * p(t)                // For all other actions
     *
     * The failure updates are in the form:
     *
     *     p(t + 1) = (1 - b) * p(t)                 // For the action taken
     *     p(t + 1) = b / (|A| - 1) + (1 - b) * p(t) // For all other actions
     *
     */
    class LRPPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * These two parameters control learning. The 'a' parameter
             * controls the learning when an action results in a success, while
             * 'b' the learning during a failure.
             *
             * Setting 'b' to zero results in an algorithm called "Linear
             * Reward-Inaction", while setting 'a' == 'b' results in the
             * "Linear Reward-Penalty" algorithm. Setting 'a' to zero results
             * in the "Linear Inaction-Penalty" algorithm.
             *
             * By default the policy is initialized with uniform distribution.
             *
             * @param A The size of the action space.
             * @param a The learning parameter on successful actions.
             * @param b The learning parameter on failed actions.
             */
            LRPPolicy(size_t A, double a, double b = 0.0);

            /**
             * @brief This function updates the LRP policy based on the result of the action.
             *
             * Note that LRP works with binary rewards: either the action
             * worked or it didn't.
             *
             * Environments where rewards are in R can be simulated: scale all
             * rewards to the [0,1] range, and stochastically obtain a success
             * with a probability equal to the reward. The result is equivalent
             * to the original reward function.
             *
             * @param a The action taken.
             * @param result Whether the action taken was a success, or not.
             */
            void stepUpdateP(size_t a, bool result);

            /**
             * @brief This function chooses an action, following the policy distribution.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * @param a The selected action.
             *
             * @return The probability of taking the selected action.
             */
            virtual double getActionProbability(const size_t & a) const override;

            /**
             * @brief This function sets the a parameter.
             *
             * The a parameter determines the amount of learning on successful actions.
             *
             * @param a The new a parameter.
             */
            void setAParam(double a);

            /**
             * @brief This function will return the currently set a parameter.
             *
             * @return The currently set a parameter.
             */
            double getAParam() const;

            /**
             * @brief This function sets the b parameter.
             *
             * The b parameter determines the amount of learning on losing actions.
             *
             * @param a The new b parameter.
             */
            void setBParam(double b);

            /**
             * @brief This function will return the currently set b parameter.
             *
             * @return The currently set b parameter.
             */
            double getBParam() const;

            /**
             * @brief This function returns a vector containing all probabilities of the policy.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Vector getPolicy() const override;

        private:
            double a_, invB_, divB_;
            Vector policy_;
    };
}

#endif
