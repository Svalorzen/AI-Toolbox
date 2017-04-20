#ifndef AI_TOOLBOX_FACTOREDMDP_XXX_ALGORITHM_HEADER_FILE
#define AI_TOOLBOX_FACTOREDMDP_XXX_ALGORITHM_HEADER_FILE

#include <AIToolbox/FactoredMDP/Types.hpp>
#include <AIToolbox/FactoredMDP/FactorGraph.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        class XXXAlgorithm {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This constructor needs to know in advance the groups of
                 * agents that need to collaboratively cooperate in order to
                 * reach their goal. This is converted in a simple Q-Table
                 * containing the learned averages for those groups.
                 *
                 * Note: each group must be unique! Not only duplicates are
                 * ignored here, but this must also be taken into consideration
                 * when producing the rewards. Duplicate factors' rewards must
                 * be summed together before reporting them to this class.
                 *
                 * @param a The factored action space of the problem.
                 * @param dependenciesAndRanges A list of [range, [agent, ..], ..] for each subgroup of connected agents.
                 */
                XXXAlgorithm(Action a, const std::vector<std::pair<double, std::vector<size_t>>> & dependenciesAndRanges);

                /**
                 * @brief This function updates the learning process from the previous action and reward.
                 *
                 * This function automatically increases the current internal timestep counter.
                 *
                 * The rewards must be in the same order as the groups were
                 * given in the constructor.
                 *
                 * @param a The action performed in the previous timestep.
                 * @param rew The rewards obtained in the previous timestep, one per agent group.
                 *
                 * @return The new optimal action to be taken at the next timestep.
                 */
                Action stepUpdateQ(const Action & a, const Rewards & rew);

                /**
                 * @brief This function returns the currently learned policy in the form of QFunctionRules.
                 *
                 * These rules skip the exploration part, to allow the creation
                 * of a greedy policy greedily with respect to the learned
                 * QFunction (since otherwise this algorithm would forever
                 * explore).
                 *
                 * @return The learned QFunctionRules.
                 */
                std::vector<QFunctionRule> toRules() const;

                /**
                 * @brief This function returns the currently set internal timestep.
                 *
                 * @return The currently set internal timestep.
                 */
                unsigned getTimestep() const;

                /**
                 * @brief This function sets the internal timestep.
                 *
                 * This function normally does not need to be called since
                 * stepUpdateQ() automatically increases the timestep. This
                 * function is provided if that functionality is not enough for
                 * some reason.
                 *
                 * Keep in mind that stepUpdateQ will first increase the
                 * internal timestep, then use the increased one. So to signal
                 * that this is going to be the first timestep, the input
                 * should be 0, and so on.
                 *
                 * @param t The new internal timestep.
                 */
                void setTimestep(unsigned t);

            private:
                /**
                 * @brief This factor contains averages and ranges for each group of agents.
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
                    double rangeSquared;
                };

                /// The action space
                Action A;
                /// The current timestep, used to compute logtA
                unsigned timestep_;
                /// The graph containing the averages and ranges for the agents.
                FactorGraph<Factor> graph_;
                /// Precomputed logA since it won't change.
                double logA_;
        };
    }
}

#endif
