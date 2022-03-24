#ifndef AI_TOOLBOX_BANDIT_TOP_TWO_THOMPSON_SAMPLING_POLICY_HEADER_FILE
#define AI_TOOLBOX_BANDIT_TOP_TWO_THOMPSON_SAMPLING_POLICY_HEADER_FILE

#include <AIToolbox/Bandit/Types.hpp>
#include <AIToolbox/Bandit/Experience.hpp>
#include <AIToolbox/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/Bandit/Policies/ThompsonSamplingPolicy.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class implements the top-two Thompson sampling policy.
     *
     * This class uses the Student-t distribution to model normally-distributed
     * rewards with unknown mean and variance. As more experience is gained,
     * each distribution becomes a Normal which models the mean of its
     * respective arm.
     *
     * The top-two Thompson sampling policy is designed to be used in a pure
     * exploration setting. In other words, we wish to discover the best arm in
     * the shortest possible time, without the need to minimize regret while
     * doing so. This last part is the key difference to many bandit
     * algorithms, that try to exploit their knowledge more and more as time
     * goes on.
     *
     * The way this works is by focusing arm pulls on the currently estimated
     * top two arms, since those are the most likely to contend for the "title"
     * of best arm. The two top arms are estimated using Thompson sampling. We
     * first sample a first best action, and then, if needed, we keep sampling
     * until a new, different best action is sampled.
     *
     * We either take the first action sampled with probability beta, or the
     * other with probability 1-beta.
     */
    class TopTwoThompsonSamplingPolicy : public PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param exp The Experience we learn from.
             * @param beta The probability of playing the first sampled best action instead of the second sampled best.
             */
            TopTwoThompsonSamplingPolicy(const Experience & exp, double beta);

            /**
             * @brief This function chooses an action using top-two Thompson sampling.
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
    };
}

#endif
