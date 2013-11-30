#ifndef AI_TOOLBOX_EPSILONPOLICY_HEADER_FILE
#define AI_TOOLBOX_EPSILONPOLICY_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>

namespace AIToolbox {
    /**
     * @brief This class is a policy wrapper for epsilon action choice. 
     * 
     * This class is used to wrap already existing policies to implement
     * automatic exploratory behaviour (e.g. epsilon-greedy policies).
     * 
     * Please note that to obtain an epsilon-greedy policy the wrapped
     * policy needs to already be greedy with respect to the model.
     */
    class EpsilonPolicy : PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor saves the input policy and the epsilon
             * parameter for later use.
             * 
             * The epsilon parameter must be >= 0.0 and <= 1.0,
             * otherwise the constructor will throw an std::runtime_error.
             *
             * @param p
             * @param epsilon
             */
            EpsilonPolicy(const PolicyInterface & p, double epsilon = 0.9);

            /**
             * @brief This function chooses a random action for state s, following the policy distribution and epsilon.
             * 
             * This function has a probability of (1 - epsilon) of selecting
             * a random action. Otherwise, it selects an action according
             * to the distribution specified by the wrapped policy.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(size_t s) const;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             * 
             * This function takes into account parameter epsilon
             * while computing the final probability.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(size_t s, size_t a) const;

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter must be >= 0.0 and <= 1.0,
             * otherwise the function will do nothing.
             *
             * @param e The new epsilon parameter.
             */
            void setEpsilon(double e);

            /**
             * @brief This function will return the currently set epsilon parameter.
             *
             * @return The currently set epsilon parameter.
             */
            double getEpsilon() const;

        protected:
            const PolicyInterface & policy_;
            double epsilon_;

            // Used to sampled random actions
            mutable std::uniform_int_distribution<size_t> randomDistribution_;
    };
}

#endif
