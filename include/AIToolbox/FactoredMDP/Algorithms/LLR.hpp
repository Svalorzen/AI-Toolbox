#ifndef AI_TOOLBOX_FACTOREDMDP_LEARNING_WITH_LINEAR_REWARDS_HEADER_FILE
#define AI_TOOLBOX_FACTOREDMDP_LEARNING_WITH_LINEAR_REWARDS_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/FactoredMDP/Types.hpp>
#include <AIToolbox/FactoredMDP/FactorGraph.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        /**
         * @brief This class represents the Learning with Linear Rewards algorithm.
         *
         * The LLR algorithm is used on multi-armed bandits, where multiple
         * actions can be taken at the same time.
         *
         * This algorithm, as described in the paper, is extremely flexible as
         * it both allows multiple actions to be taken at each timestep, while
         * also leaving space for any algorithm which is able to solve the
         * action maximization selection problem. This is possible since the
         * action space can be arbitrarily restricted.
         *
         * This means that creating an actual generic algorithm out of the
         * paper is pretty hard as it would have to be able to be passed any
         * algorithm and use it. We chose not to do it here.
         *
         * Here we implement a simple version where a single, factored action
         * is allowed, and we use VE to solve the action selection problem.
         * This pretty much results in simply solving VE with UCB1 weights,
         * together with some learning.
         */
        class LLR {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * In order to keep track of each partial action's averages and
                 * counts, we need to know which factors are actually dependent
                 * on each other.
                 *
                 * So suppose we have a three-factored action space {1,2,3},
                 * and two local reward functions using factors {0,1}, and
                 * {1,2}. Then {{0,1}, {1,2}} is going to be the dependency
                 * parameter.
                 *
                 * @param a The action space.
                 * @param dependencies The dependencies in the problem.
                 */
                LLR(Action a, const std::vector<Factors> & dependencies);

                /**
                 * @brief This function updates the learning process from the previous action and reward.
                 *
                 * Note that the rewards parameter is going to have as many
                 * elements as the number of local payoff functions passed as
                 * input in the constructor.
                 *
                 * @param a The action taken in the previous step.
                 * @param r The rewards obtained in the previous step.
                 *
                 * @return The optimal action to take at the next timestep.
                 */
                Action stepUpdateQ(const Action & a, const Rewards & r);

            private:
                /**
                 * @brief This factor contains averages and counts for each group of agents.
                 */
                struct Factor {
                    struct Average {
                        double value = 0.0;
                        unsigned count = 0;
                    };
                    /**
                     * @brief The Q-Table for this factor's agents.
                     *
                     * Since we don't know in advance what the dimensionality
                     * of this Q-Table may be, we simply create a
                     * one-dimensional vector and we automatically compute its
                     * indices as if it was a multi-dimensional array.
                     */
                    std::vector<Average> averages;
                };

                /// The action space
                Action A;
                /// The number of actions allowed at any one time (always 1)
                unsigned L;
                /// The current timestep, to compute the UCB1 value
                unsigned timestep_;
                /// The graph containing the averages and ranges for the agents.
                FactorGraph<Factor> graph_;
                /// This counter is used to know when to start using UCB1 and stop doing random actions.
                unsigned missingExplorations_;
        };
    }
}

#endif
