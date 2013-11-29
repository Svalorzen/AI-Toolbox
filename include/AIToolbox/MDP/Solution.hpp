#ifndef AI_TOOLBOX_MDP_SOLUTION_HEADER_FILE
#define AI_TOOLBOX_MDP_SOLUTION_HEADER_FILE

#include <AIToolbox/Policy.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        class Solution {
            public:
                Solution(size_t s, size_t a);

                Policy &          getPolicy();
                ValueFunction &   getValueFunction();
                QFunction &       getQFunction();

                const Policy &          getPolicy()             const;
                const ValueFunction &   getValueFunction()      const;
                const QFunction &       getQFunction()          const;

                size_t getGreedyQAction(size_t s) const;

                /**
                 * @brief This function returns the number of states of the world.
                 *
                 * @return The total number of states.
                 */
                size_t getS() const;

                /**
                 * @brief This function returns the number of available actions to the agent.
                 *
                 * @return The total number of actions.
                 */
                size_t getA() const;
            private:
                size_t S, A;

                QFunction q_;
                ValueFunction v_;
                Policy policy_;
        };
    }
}

#endif
