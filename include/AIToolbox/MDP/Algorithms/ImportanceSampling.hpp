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

            using Parent::Parent;

            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double, size_t maxA) const {
                const auto baseProb = (1.0 - exploration_) / A;
                return (baseProb + (maxA == a) * exploration_) / behaviour_.getActionProbability(s, a);
            }
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

            using Parent::Parent;

            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double) const {
                return target_.getActionProbability(s, a) / behaviour_.getActionProbability(s, a);
            }
    };
}

#endif
