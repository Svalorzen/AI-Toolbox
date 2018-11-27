#ifndef AI_TOOLBOX_MDP_IMPORTANCE_SAMPLING_HEADER_FILE
#define AI_TOOLBOX_MDP_IMPORTANCE_SAMPLING_HEADER_FILE

#include <AIToolbox/MDP/Algorithms/Utils/OffPolicyTemplate.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class implements off-policy control via importance sampling.
     *
     * \sa ImportanceSamplingEvaluation
     */
    class ImportanceSampling : public OffPolicyControl<ImportanceSampling> {
        public:
            using Parent = OffPolicyControl<ImportanceSampling>;

            /**
             * @brief Basic constructor.
             *
             * @param behaviour Behaviour policy
             * @param discount Discount for the problem.
             * @param alpha Learning rate parameter.
             * @param tolerance Trace cutoff parameter.
             * @param epsilon The epsilon of the implied target greedy epsilon policy.
             */
            ImportanceSampling(const PolicyInterface & behaviour, const double discount = 1.0,
                     const double alpha = 0.1, const double tolerance = 0.001, const double epsilon = 0.1) :
                    Parent(behaviour.getS(), behaviour.getA(), discount, alpha, tolerance, epsilon),
                    behaviour_(behaviour) {}

        private:
            friend Parent;
            /**
             * @brief This function returns the trace discount for the learning.
             *
             * This function returns the ratio between the assumed epsilon-greedy policy and the behaviour policy.
             */
            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double, const size_t maxA) const {
                const auto prob = epsilon_ / A + (a == maxA) * (1.0 - epsilon_);
                return prob / behaviour_.getActionProbability(s, a);
            }

            const PolicyInterface & behaviour_;
    };

    /**
     * @brief This class implements off-policy evaluation via importance sampling.
     *
     * This off policy algorithm weights the traces based on the ratio of the
     * likelyhood of the target policy vs the behaviour policy.
     *
     * The idea behind this is that if an action is very unlikely to be taken
     * by the behaviour with respect to the target, then we should count it
     * more, as if to "simulate" the returns we'd get when acting with the
     * target policy.
     *
     * On the other side, if an action is very likely to be taken by the
     * behaviour policy with respect to the target, we're going to count it
     * less, as we're probably going to see this action picked a lot more times
     * than what we'd have done with the target.
     *
     * While this method is correct in theory, in practice it suffers from an
     * incredibly high, if possibly infinite, variance. What happens is if you
     * get a sequence of lucky (or unlucky) action choices, the traces get
     * either cut or, even worse, get incredibly high valued, which skews the
     * results quite a lot.
     */
    class ImportanceSamplingEvaluation : public OffPolicyEvaluation<ImportanceSamplingEvaluation> {
        public:
            using Parent = OffPolicyEvaluation<ImportanceSamplingEvaluation>;

            /**
             * @brief Basic constructor.
             *
             * @param target Target policy.
             * @param behaviour Behaviour policy
             * @param discount Discount for the problem.
             * @param alpha Learning rate parameter.
             * @param tolerance Trace cutoff parameter.
             */
            ImportanceSamplingEvaluation(const PolicyInterface & target, const PolicyInterface & behaviour,
                               const double discount, const double alpha, const double tolerance) :
                    Parent(target, discount, alpha, tolerance),
                    behaviour_(behaviour) {}

        private:
            friend Parent;
            /**
             * @brief This function returns the trace discount for the learning.
             *
             * This function returns the ratio between the target and the behaviour policy.
             */
            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double) const {
                return target_.getActionProbability(s, a) / behaviour_.getActionProbability(s, a);
            }

            const PolicyInterface & behaviour_;
    };
}

#endif
