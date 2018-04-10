#ifndef AI_TOOLBOX_FACTORED_BANDIT_Q_GREEDY_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_Q_GREEDY_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>
#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredContainer.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class models a greedy policy through a QFunction.
     *
     * This class allows you to select effortlessly the best greedy actions
     * from a given list of QFunctionRules. In order to compute the best
     * action, or a given action probability the QGreedyPolicy must run
     * VariableElimination on the stored rules, so the process can get a
     * bit expensive.
     */
    class QGreedyPolicy : public PolicyInterface {
        public:
            using Base = PolicyInterface;

            /**
             * @brief Basic constructor.
             *
             * @param a The number of actions available to the agent.
             * @param q The QFunction this policy is linked with.
             */
            QGreedyPolicy(Action a, const FactoredContainer<QFunctionRule> & q);

            /**
             * @brief This function chooses the greediest action for state s.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * @param a The selected action.
             *
             * @return This function returns 1 if a is equal to the greediest action, and 0 otherwise.
             */
            virtual double getActionProbability(const Action & a) const override;

        private:
            const FactoredContainer<QFunctionRule> & q_;
    };
}

#endif
