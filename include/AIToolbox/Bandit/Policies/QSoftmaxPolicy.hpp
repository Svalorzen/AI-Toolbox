#ifndef AI_TOOLBOX_BANDIT_Q_SOFTMAX_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_Q_SOFTMAX_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Types.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class models a simple softmax policy.
     *
     * This class always selects the greediest action with respect to the
     * already obtained experience.
     */
    class QSoftmaxPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             */
            QSoftmaxPolicy(const QFunction & q, double t);

            /**
             * @brief This function chooses the greediest action.
             *
             * If multiple actions would be equally as greedy, a random one
             * is returned.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * If multiple greedy actions exist, this function returns the
             * correct probability of picking each one, since we return a
             * random one with sampleAction().
             *
             * @param a The selected action.
             *
             * @return This function returns 0 if the action is not greedy, and 1/the number of greedy actions otherwise.
             */
            virtual double getActionProbability(const size_t & a) const override;

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

            /**
             * @brief This function returns a vector containing all probabilities of the policy.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             */
            virtual Vector getPolicy() const override;

        private:
            double temperature_;
            const QFunction & q_;
            // To avoid reallocating a vector every time for sampling.
            mutable std::vector<size_t> bestActions_;
    };
}

#endif
