#ifndef AI_TOOLBOX_MDP_TREE_BACKUP_L_HEADER_FILE
#define AI_TOOLBOX_MDP_TREE_BACKUP_L_HEADER_FILE

#include <AIToolbox/MDP/Algorithms/Utils/OffPolicyTemplate.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class implements off-policy control via Tree Backup(lambda).
     *
     * \sa TreeBackupLEvaluation
     */
    class TreeBackupL : public OffPolicyControl<TreeBackupL> {
        public:
            using Parent = OffPolicyControl<TreeBackupL>;

            /**
             * @brief Basic constructor.
             *
             * @param behaviour Behaviour policy
             * @param lambda Lambda trace parameter.
             * @param epsilon The epsilon of the implied target greedy epsilon policy.
             * @param discount Discount for the problem.
             * @param alpha Learning rate parameter.
             * @param tolerance Trace cutoff parameter.
             */
            TreeBackupL(const PolicyInterface & behaviour, const double lambda, const double epsilon = 0.1,
                        const double discount = 1.0, const double alpha = 0.1, const double tolerance = 0.001) :
                    Parent(behaviour, epsilon, discount, alpha, tolerance)
            {
                setLambda(lambda);
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
            friend Parent;
            /**
             * @brief This function returns the trace discount for the learning.
             */
            double getTraceDiscount(const size_t, const size_t a, const size_t, const double, const size_t maxA) const {
                const auto prob = epsilon_ / A + (a == maxA) * (1.0 - epsilon_);
                return lambda_ * prob;
            }

            double lambda_;
    };

    /**
     * @brief This class implements off-policy evaluation via Tree Backup(lambda).
     *
     * This algorithm tries to avoid the infinite variance problem that
     * ImportanceSampling has, by multiplying the traces by just the target
     * policy probability. It additionally uses the lambda parameter to further
     * tune their length.
     *
     * While it succeeds in its intent, it tends to cut traces short. This
     * happens since all actions taken by a policy have a <= 1 probability of
     * being picked, which generally shortens the trace. While not overall a
     * problem, this is inefficient in case the behaviour and target policies
     * are very similar.
     */
    class TreeBackupLEvaluation : public OffPolicyEvaluation<TreeBackupLEvaluation> {
        public:
            using Parent = OffPolicyEvaluation<TreeBackupLEvaluation>;

            /**
             * @brief Basic constructor.
             *
             * @param target Target policy.
             * @param behaviour Behaviour policy
             * @param lambda Lambda trace parameter.
             * @param discount Discount for the problem.
             * @param alpha Learning rate parameter.
             * @param tolerance Trace cutoff parameter.
             */
            TreeBackupLEvaluation(const PolicyInterface & target, const PolicyInterface & behaviour,
                                  const double lambda, const double discount, const double alpha, const double tolerance) :
                    Parent(target, behaviour, discount, alpha, tolerance)
            {
                setLambda(lambda);
            }

            /**
             * @brief This function returns the trace discount for the learning.
             */
            double getTraceDiscount(const size_t s, const size_t a, const size_t, const double) const {
                return lambda_ * target_.getActionProbability(s, a);
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
