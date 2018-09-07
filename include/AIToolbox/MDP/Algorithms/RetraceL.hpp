#ifndef AI_TOOLBOX_MDP_RETRACE_L_HEADER_FILE
#define AI_TOOLBOX_MDP_RETRACE_L_HEADER_FILE

#include <AIToolbox/MDP/Algorithms/Utils/OffPolicyTemplate.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class implements off-policy control via Retrace(lambda).
     *
     * \sa RetraceLEvaluation
     */
    class RetraceL : public OffPolicyControl<RetraceL> {
        public:
            using Parent = OffPolicyControl<RetraceL>;

            RetraceL(const PolicyInterface & behaviour, const double lambda, const double exploration = 0.9,
                     const double discount = 1.0, const double alpha = 0.1, const double epsilon = 0.001) :
                    Parent(behaviour, exploration, discount, alpha, epsilon)
            {
                setLambda(lambda);
            }

            /**
             * @brief This function returns the trace discount for the learning.
             */
            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double, const size_t maxA) const {
                const auto prob = (1.0 - exploration_) / A + (a == maxA) * exploration_;
                return lambda_ * std::min(1.0, prob / behaviour_.getActionProbability(s, a));
            }

            /**
             * @brief This function sets the new lambda parameter.
             *
             * The lambda parameter must be >= 0.0 and <= 1.0, otherwise the
             * function will throw an std::invalid_argument.
             *
             * @param l The new lambda parameter.
             */
            void setLambda(double l) {
                if ( l < 0.0 || l > 1.0 ) throw std::invalid_argument("Lambda parameter must be in [0,1]");
                lambda_ = l;
            }

            /**
             * @brief This function returns the currently set lambda parameter.
             */
            double getLambda() const { return lambda_; }

        private:
            double lambda_;
    };

    /**
     * @brief This class implements off-policy evaluation via Retrace(lambda).
     *
     * This algorithm tries to get all advantages from ImportanceSampling, QL
     * and TreeBackupL. The idea is to use the lambda parameter to tune the
     * traces, but at the same time use the ratio between target and behaviour
     * policies in order to make the most out of the available data.
     *
     * To avoid the variance problem of ImportanceSampling though, it imposes a
     * ceiling on the ratio: if too high it is pinned to 1. This still
     * leverages the data, but makes variance much less of a problem, since now
     * traces are bound to decrease over time.
     */
    class RetraceLEvaluation : public OffPolicyEvaluation<RetraceLEvaluation> {
        public:
            using Parent = OffPolicyEvaluation<RetraceLEvaluation>;

            /**
             * @brief Basic constructor.
             *
             * @param target Target policy.
             * @param behaviour Behaviour policy
             * @param lambda Lambda trace parameter.
             * @param discount Discount for the problem.
             * @param alpha Learning rate parameter.
             * @param epsilon Trace cutoff parameter.
             */
            RetraceLEvaluation(const PolicyInterface & target, const PolicyInterface & behaviour,
                               const double lambda, const double discount, const double alpha, const double epsilon) :
                    Parent(target, behaviour, discount, alpha, epsilon)
            {
                setLambda(lambda);
            }

            /**
             * @brief This function returns the trace discount for the learning.
             */
            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double) const {
                return lambda_ * std::min(1.0, target_.getActionProbability(s, a) / behaviour_.getActionProbability(s, a));
            }

            /**
             * @brief This function sets the new lambda parameter.
             *
             * The lambda parameter must be >= 0.0 and <= 1.0, otherwise the
             * function will throw an std::invalid_argument.
             *
             * @param l The new lambda parameter.
             */
            void setLambda(double l) {
                if ( l < 0.0 || l > 1.0 ) throw std::invalid_argument("Lambda parameter must be in [0,1]");
                lambda_ = l;
            }

            /**
             * @brief This function returns the currently set lambda parameter.
             */
            double getLambda() const { return lambda_; }

        private:
            double lambda_;
    };
}

#endif

