#ifndef AI_TOOLBOX_BANDIT_THOMPSON_SAMPLING_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_THOMPSON_SAMPLING_POLICY_HEADER_FILE

#include <random>

#include <AIToolbox/Bandit/Types.hpp>
#include <AIToolbox/Bandit/Experience.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class models a Thompson sampling policy.
     *
     * This class uses the Student-t distribution to model normally-distributed
     * rewards with unknown mean and variance. As more experience is gained,
     * each distribution becomes a Normal which models the mean of its
     * respective arm.
     */
    class ThompsonSamplingPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param exp The Experience we learn from.
             */
            ThompsonSamplingPolicy(const Experience & exp);

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

            /**
             * @brief This function returns a vector containing all probabilities of the policy.
             *
             * Ideally this function can be called only when there is a
             * repeated need to access the same policy values in an
             * efficient manner.
             *
             * WARNING: This can be really expensive, as it does pretty much
             * the same work as getActionProbability(). It shouldn't be slower
             * than that call though, so if you do need the overall policy,
             * do call this method.
             */
            virtual Vector getPolicy() const override;

        private:
            const Experience & exp_;
    };
}

#endif
