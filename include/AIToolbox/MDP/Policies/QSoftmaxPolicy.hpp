#ifndef AI_TOOLBOX_MDP_Q_SOFTMAX_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_Q_SOFTMAX_POLICY_HEADER_FILE

#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class models a softmax policy through a QFunction.
     *
     * A softmax policy is a policy that selects actions based on their
     * expected reward: the more advantageous an action seems to be, the more
     * probable its selection is. There are many ways to implement a softmax
     * policy, this class implements selection using the most common method of
     * sampling from a Boltzmann distribution.
     *
     * As the epsilon-policy, this type of policy is useful to force the agent
     * to explore an unknown model, in order to gain new information to refine
     * it and thus gain more reward.
     */
    class QSoftmaxPolicy : public QPolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * The temperature parameter must be >= 0.0
             * otherwise the constructor will throw an std::invalid_argument.
             *
             * @param q The QFunction this policy is linked with.
             * @param temperature The parameter that controls the amount of exploration.
             */
            QSoftmaxPolicy(const QFunction & q, double t = 1.0);

            /**
             * @brief This function chooses an action for state s with probability dependent on value.
             *
             * This class implements softmax through the Boltzmann
             * distribution. Thus an action will be chosen with
             * probability:
             *
             * \f[
             *      P(a) = \frac{e^{(Q(s,a)/t)})}{\sum_b{e^{(Q(s,b)/t)}}}
             * \f]
             *
             * where t is the temperature. This value is not cached anywhere, so
             * continuous sampling may not be extremely fast.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(const size_t & s) const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * \sa sampleAction(const size_t & s);
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the specified action in the specified state.
             */
            virtual double getActionProbability(const size_t & s, const size_t & a) const override;

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
            virtual Matrix2D getPolicy() const override;

            /**
             * @brief This function sets the temperature parameter.
             *
             * The temperature parameter determines the amount of
             * exploration this policy will enforce when selecting actions.
             * Following the Boltzmann distribution, as the temperature
             * approaches infinity all actions will become equally
             * probable. On the opposite side, as the temperature
             * approaches zero, action selection will become completely
             * greedy.
             *
             * The temperature parameter must be >= 0.0 otherwise the
             * function will do throw std::invalid_argument.
             *
             * @param t The new temperature parameter.
             */
            void setTemperature(double t);

            /**
             * @brief This function will return the currently set temperature parameter.
             *
             * @return The currently set temperature parameter.
             */
            double getTemperature() const;

        private:
            double temperature_;
            // To avoid reallocating a vector every time for sampling.
            mutable std::vector<size_t> bestActions_;
            mutable Vector vbuffer_;
    };
}

#endif
