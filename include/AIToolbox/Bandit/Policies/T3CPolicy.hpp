#ifndef AI_TOOLBOX_BANDIT_T3C_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_T3C_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Types.hpp>
#include <AIToolbox/Bandit/Experience.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class implements the T3C sampling policy.
     *
     * This class assumes that the rewards of all bandit arms are normally
     * distributed, with all arms having the same variance.
     *
     * T3C was designed as a replacement for TopTwoThompsonSamplingPolicy. The
     * main idea is that, when we want to pull the estimated *second* best arm,
     * instead of having to resample the arm means until a new unique contender
     * appears, we can deterministically compute that contender using a measure
     * of distance between the distributions of the arms.
     *
     * This allows the algorithm to keep the computational costs low even after
     * many pulls, while TopTwoThompsonSamplingPolicy tends to degrade in
     * performance as time passes (as resampling is less and less likely to
     * generate a unique second best contender).
     */
    class T3CPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param exp The Experience we learn from.
             * @param beta The probability of playing the first sampled best action instead of the second sampled best.
             * @param var The known variance of all arms.
             */
            T3CPolicy(const Experience & exp, double beta, double var);

            /**
             * @brief This function chooses an action using T3CPolicy.
             *
             * @return The chosen action.
             */
            virtual size_t sampleAction() const override;

            /**
             * @brief This function returns the most likely best action until this point.
             *
             * @return The most likely best action.
             */
            size_t recommendAction() const;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * WARNING: The only way to compute the true probability of
             * selecting the input action is via empirical sampling.  we simply
             * call sampleAction() a lot and return an approximation of the
             * times the input action was actually selected. This makes this
             * function very very SLOW. Do not call at will!!
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

            /**
             * @brief This function returns a reference to the underlying Experience we use.
             *
             * @return The internal Experience reference.
             */
            const Experience & getExperience() const;

        private:
            ThompsonSamplingPolicy policy_;
            double beta_;
            double var_;
    };
}

#endif

