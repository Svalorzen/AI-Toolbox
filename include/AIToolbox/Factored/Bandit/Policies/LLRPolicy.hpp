#ifndef AI_TOOLBOX_FACTORED_BANDIT_LEARNING_WITH_LINEAR_REWARDS_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_LEARNING_WITH_LINEAR_REWARDS_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Bandit/Experience.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>
#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents the Learning with Linear Rewards algorithm.
     *
     * The LLR algorithm is used on multi-armed bandits, where multiple
     * actions can be taken at the same time.
     *
     * This algorithm, as described in the paper, is extremely flexible as
     * it both allows multiple actions to be taken at each timestep, while
     * also leaving space for any algorithm which is able to solve the
     * action maximization selection problem. This is possible since the
     * action space can be arbitrarily restricted.
     *
     * This means that creating an actual generic algorithm out of the
     * paper is pretty hard as it would have to be able to be passed any
     * algorithm and use it. We chose not to do it here.
     *
     * Here we implement a simple version where a single, factored action
     * is allowed, and we use VE to solve the action selection problem.
     * This pretty much results in simply solving VE with UCB1 weights,
     * together with some learning.
     */
    class LLRPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param exp The Experience we learn from.
             */
            LLRPolicy(const Experience & exp);

            /**
             * @brief This function selects an action using LLR.
             *
             * We construct a VE process, where for each entry we compute
             * independently its exploration bonus. This is imprecise because
             * we end up overestimating the bonus and over-exploring.
             *
             * For improved alternatives, look at MAUCE or ThompsonSamplingPolicy.
             *
             * \sa MAUCE
             * \sa ThompsonSamplingPolicy
             *
             * @return The optimal action to take at the next timestep.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * As sampleAction() is deterministic, we simply run it to check
             * that the Action it returns is equal to the one passed as input.
             *
             * @param a The selected action.
             *
             * @return This function returns an approximation of the probability of choosing the input action.
             */
            virtual double getActionProbability(const Action & a) const override;

            /**
             * @brief This function returns the Experience we use to learn.
             *
             * @return The underlying Experience.
             */
            const Experience & getExperience() const;

        private:
            /// The Experience containing all averages and counts for all local joint actions.
            const Experience & exp_;
            /// The number of actions allowed at any one time (always 1)
            unsigned L;
    };
}

#endif
