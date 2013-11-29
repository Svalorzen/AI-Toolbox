#ifndef AI_TOOLBOX_MDP_SOLUTION_HEADER_FILE
#define AI_TOOLBOX_MDP_SOLUTION_HEADER_FILE

#include <AIToolbox/Policy.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class contains all relevant MDP information of a solved Model.
         */
        class Solution {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 */
                Solution(size_t s, size_t a);

                /**
                 * @brief This function gives access to the internal Policy to be modified.
                 *
                 * @return A reference to the internal Policy.
                 */
                Policy &          getPolicy();
                /**
                 * @brief This function gives access to the internal ValueFunction to be modified.
                 *
                 * @return A reference to the internal ValueFunction.
                 */
                ValueFunction &   getValueFunction();
                /**
                 * @brief This function gives access to the internal QFunction to be modified.
                 *
                 * @return A reference to the internal QFunction.
                 */
                QFunction &       getQFunction();

                /**
                 * @brief This function gives access to the internal Policy to be inspected.
                 *
                 * @return A const reference to the internal Policy.
                 */
                const Policy &          getPolicy()             const;
                /**
                 * @brief This function gives access to the internal ValueFunction to be inspected.
                 *
                 * @return A const reference to the internal ValueFunction.
                 */
                const ValueFunction &   getValueFunction()      const;
                /**
                 * @brief This function gives access to the internal QFunction to be inspected.
                 *
                 * @return A const reference to the internal QFunction.
                 */
                const QFunction &       getQFunction()          const;

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
