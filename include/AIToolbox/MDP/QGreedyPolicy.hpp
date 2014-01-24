#ifndef AI_TOOLBOX_MDP_Q_GREEDY_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_Q_GREEDY_POLICY_HEADER_FILE

#include <AIToolbox/MDP/QPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class models a greedy policy through a QFunction.
         * 
         */
        class QGreedyPolicy : public QPolicyInterface {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param q The QFunction this policy is linked with.
                 */
                QGreedyPolicy(const QFunction & q);

                /**
                 * @brief This function chooses the greediest action for state s.
                 *
                 * @param s The sampled state of the policy.
                 *
                 * @return The chosen action.
                 */
                virtual size_t sampleAction(size_t s) const override;

                /**
                 * @brief This function returns the probability of taking the specified action in the specified state.
                 * 
                 * @param s The selected state.
                 * @param a The selected action.
                 *
                 * @return This function returns 1 if a is equal to the greediest action, and 0 otherwise.
                 */
                virtual double getActionProbability(size_t s, size_t a) const override;
        };
    }
}

#endif
