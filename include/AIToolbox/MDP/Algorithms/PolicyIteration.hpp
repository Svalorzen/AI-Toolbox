#ifndef AI_TOOLBOX_MDP_POLICY_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_ITERATION_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/MDP/Algorithms/Utils/PolicyEvaluation.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the Policy Iteration algorithm.
     *
     * This algorithm begins with an arbitrary policy (random), and uses
     * the PolicyEvaluation algorithm to find out the Values for each state
     * of this policy.
     *
     * Once this is done, the policy can be improved by using a greedy
     * approach towards the QFunction found. The new policy is then newly
     * evaluated, and the process repeated.
     *
     * When the policy does not change anymore, it is guaranteed to be
     * optimal, and the found QFunction is returned.
     */
    class PolicyIteration {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param horizon The horizon parameter to use during the PolicyEvaluation phase.
             * @param tolerance The tolerance parameter to use during the PolicyEvaluation phase.
             */
            PolicyIteration(unsigned horizon, double tolerance = 0.001);

            /**
             * @brief This function applies policy iteration on an MDP to solve it.
             *
             * The algorithm is constrained by the currently set parameters.
             *
             * @param m The MDP that needs to be solved.
             * @return The QFunction of the optimal policy found.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            QFunction operator()(const M & m);

            /**
             * @brief This function sets the tolerance parameter.
             *
             * The tolerance parameter must be >= 0 or the function will throw.
             */
            void setTolerance(double t);

            /**
             * @brief This function sets the horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function returns the currently set tolerance parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function returns the currently set horizon parameter.
             */
            unsigned getHorizon() const;

        private:
            unsigned horizon_;
            double tolerance_;
    };

    template <typename M, typename>
    QFunction PolicyIteration::operator()(const M & m) {
        const auto S = m.getS();
        const auto A = m.getA();

        PolicyEvaluation<M> eval(m, horizon_, tolerance_);

        auto qfun = makeQFunction(m.getS(), m.getA());
        QGreedyPolicy p(qfun);
        auto matrix = p.getPolicy();

        {
nextLoop:
            auto [bound, v, q] = eval(p);
            (void)bound;

            eval.setValues(std::move(v));
            qfun = std::move(q);

            auto newMatrix = p.getPolicy();
            for (size_t s = 0; s < S; ++s) {
                for (size_t a = 0; a < A; ++a) {
                    if (checkDifferentSmall(matrix(s,a), newMatrix(s,a))) {
                        matrix = std::move(newMatrix);
                        goto nextLoop;
                    }
                }
            }
        }
        return qfun;
    }
}

#endif
