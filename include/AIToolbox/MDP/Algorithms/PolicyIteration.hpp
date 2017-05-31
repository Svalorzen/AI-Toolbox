#ifndef AI_TOOLBOX_MDP_POLICY_ITERATION_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_ITERATION_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
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
             * @param epsilon The epsilon parameter to use during the PolicyEvaluation phase.
             */
            PolicyIteration(unsigned horizon, double epsilon = 0.001);

            /**
             * @brief This function applies policy iteration on an MDP to solve it.
             *
             * The algorithm is constrained by the currently set parameters.
             *
             * @param m The MDP that needs to be solved.
             * @return The QFunction of the optimal policy found.
             */
            template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
            QFunction operator()(const M & m);

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter must be >= 0 or the function will throw.
             */
            void setEpsilon(double e);

            /**
             * @brief This function sets the horizon parameter.
             */
            void setHorizon(unsigned h);

            /**
             * @brief This function returns the currently set epsilon parameter.
             */
            double getEpsilon() const;

            /**
             * @brief This function returns the currently set horizon parameter.
             */
            unsigned getHorizon() const;

        private:
            unsigned horizon_;
            double epsilon_;
    };

    template <typename M, typename>
    QFunction PolicyIteration::operator()(const M & m) {
        const auto S = m.getS();
        const auto A = m.getA();

        PolicyEvaluation<M> eval(m, horizon_, epsilon_);

        auto qfun = makeQFunction(m.getS(), m.getA());
        QGreedyPolicy p(qfun);
        auto table = p.getPolicy();

        {
nextLoop:
            auto solution = eval(p);

            eval.setValues(std::move(std::get<1>(solution)));
            qfun = std::move(std::get<2>(solution));

            auto newTable = p.getPolicy();
            for (size_t s = 0; s < S; ++s) {
                for (size_t a = 0; a < A; ++a) {
                    if (checkDifferentSmall(table(s,a), newTable(s,a))) {
                        table = std::move(newTable);
                        goto nextLoop;
                    }
                }
            }
        }

        return std::move(qfun);
    }
}

#endif
