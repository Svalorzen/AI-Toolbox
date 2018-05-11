#ifndef AI_TOOLBOX_BANDIT_THOMPSON_SAMPLING_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_THOMPSON_SAMPLING_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class models a Thompson sampling policy.
     *
     * This class keeps a record of the rewards obtained by each action, and
     * chooses them with a stochastic policy which is proportional to the
     * goodness of each action.
     *
     * It uses the Normal distribution in order to estimate its certainty about
     * each arm average reward. Thus, each arm is estimated through a Normal
     * distribution centered on the average for the arm, with decreasing
     * variance as more experience is gathered.
     *
     * Note that this class assumes that the reward obtained is normalized into
     * a [0,1] range (which it does not check).
     *
     * The usage of the Normal distribution best matches a Normally distributed
     * reward. Another implementation (not provided here) uses Beta
     * distributions to handle Bernoulli distributed rewards.
     */
    class ThompsonSamplingPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             */
            ThompsonSamplingPolicy(size_t A);

            /**
             * @brief This function updates the policy based on the result of the action.
             *
             * We simply keep a rolling average for each action, which we
             * update here. Each average and count will then be used as
             * parameters for the Normal distribution used to decide which
             * action to sample later.
             *
             * Note that we expect a normalized reward here in order to
             * easily compare the various actions during Normal sampling.
             *
             * @param a The action taken.
             * @param r The reward obtained, in a [0,1] range.
             */
            void stepUpdateP(size_t a, double r);

            /**
             * @brief This function chooses an action using Thompson sampling.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * WARNING: In this class the only way to compute the true
             * probability of selecting the input action is via numerical
             * integration, since we're dealing with |A| Normal random
             * variables. To avoid having to do this, we simply sample a lot
             * and return an approximation of the times the input action was
             * actually selected. This makes this function very very SLOW. Do
             * not call at will!!
             *
             * @param a The selected action.
             *
             * @return This function returns an approximation of the probability of choosing the input action.
             */
            virtual double getActionProbability(const size_t & a) const override;

        private:
            // Average reward/tries per action
            std::vector<std::pair<double, unsigned>> experience_;
    };
}

#endif

