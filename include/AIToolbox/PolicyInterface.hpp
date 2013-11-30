#ifndef AI_TOOLBOX_POLICYINTERFACE_HEADER_FILE
#define AI_TOOLBOX_POLICYINTERFACE_HEADER_FILE

#include <cstddef>
#include <random>

namespace AIToolbox {
    /**
     * @brief This class represents the base interface for policies.
     * 
     * This class represents an interface that all policies must conform to.
     * The interface is generic as different methods may have very different
     * ways to store and compute policies, and this interface simply asks
     * for a way to sample them.
     */
    class PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            PolicyInterface(size_t s, size_t a);

            /**
             * @brief This function chooses a random action for state s, following the policy distribution.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction(size_t s) const = 0;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(size_t s, size_t a) const = 0;

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            size_t getA() const;
        protected:
            size_t S, A;

            // These are mutable because sampling doesn't really change the policy
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };

}

#endif
