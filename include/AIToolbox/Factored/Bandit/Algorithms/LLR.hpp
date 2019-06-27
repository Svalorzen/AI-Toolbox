#ifndef AI_TOOLBOX_FACTORED_BANDIT_LEARNING_WITH_LINEAR_REWARDS_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_LEARNING_WITH_LINEAR_REWARDS_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/RollingAverage.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>

namespace AIToolbox::Factored::Bandit {
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
            LLR(Action a, const std::vector<PartialKeys> & dependencies);

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

            /**
             * @brief This function returns the RollingAverage learned from the data.
             *
             * These rules skip the exploration part, to allow the creation
             * of a policy using the learned QFunction (since otherwise
             * this algorithm would forever explore).
             *
             * @return The RollingAverage containing all statistics from the input data.
             */
            const RollingAverage & getRollingAverage() const;

        private:
            struct Average {
                double value = 0.0;
                unsigned count = 0;
            };

            /// The action space
            Action A;
            /// The number of actions allowed at any one time (always 1)
            unsigned L;
            /// The current timestep, to compute the UCB1 value
            unsigned timestep_;
            /// A vector containing all averages and counts for all local joint actions.
            RollingAverage averages_;
    };
}

#endif
