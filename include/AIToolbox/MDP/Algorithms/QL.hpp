#ifndef AI_TOOLBOX_MDP_QL_HEADER_FILE
#define AI_TOOLBOX_MDP_QL_HEADER_FILE

#include <AIToolbox/MDP/Algorithms/Utils/OffPolicyTemplate.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class implements off-policy control via Q(lambda).
     *
     * \sa QLEvaluation
     *
     * This method behaves as an inefficient QLearning if you set the lambda
     * parameter to zero (effectively cutting all traces), and the exploration
     * parameter to zero (forcing a perfectly greedy target policy).
     */
    class QL : public OffPolicyControl<QL> {
        public:
            using Parent = OffPolicyControl<QL>;

            QL(const PolicyInterface & behaviour, const double lambda, const double exploration = 0.9,
               const double discount = 1.0, const double alpha = 0.1, const double epsilon = 0.001) :
                    Parent(behaviour, exploration, discount, alpha, epsilon)
            {
                setLambda(lambda);
            }

            /**
             * @brief This function returns the trace discount for the learning.
             */
            double getTraceDiscount(const size_t, const size_t, const size_t, const double, const size_t) const {
                return lambda_;
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
     * @brief This class implements off-policy evaluation via Q(lambda).
     *
     * This algorithm is the off-policy equivalent of SARSAL. It scales traces
     * using the lambda parameter, but is able to work in an off-line manner.
     *
     * Unfortunately, as it does not take into account the discrepancy between
     * behaviour and target policies, it tends to work only if the two policies
     * are similar.
     *
     * Note that even if the discount does not take into account the target
     * policy, the error update is till computed using the target, and that is
     * why the method works and does not just compute the value of the
     * behaviour policy.
     */
    class QLEvaluation : public OffPolicyEvaluation<QLEvaluation> {
        public:
            using Parent = OffPolicyEvaluation<QLEvaluation>;

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
            QLEvaluation(const PolicyInterface & target, const PolicyInterface & behaviour,
                         const double lambda, const double discount, const double alpha, const double epsilon) :
                    Parent(target, behaviour, discount, alpha, epsilon)
            {
                setLambda(lambda);
            }

            /**
             * @brief This function returns the trace discount for the learning.
             */
            double getTraceDiscount(const size_t, const size_t, const size_t, const double) const {
                return lambda_;
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
