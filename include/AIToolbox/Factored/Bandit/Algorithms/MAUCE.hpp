#ifndef AI_TOOLBOX_FACTORED_BANDIT_MAUCE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_MAUCE_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/UCVE.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/RollingAverage.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class represents the Multi-Agent Upper Confidence Exploration algorithm.
     *
     * This algorithm is similar in spirit to LLR, but it performs a much more
     * sophisticated variable elimination step that includes branch-and-bound.
     *
     * It does this by knowing, via its parameters, the maximum reward range
     * for each group of interdependent agents (max possible reward minus min
     * possible reward). This allows it to estimate the uncertainty around any
     * given joint action, by keeping track for each PartialAction its upper
     * and lower bounds.
     *
     * During the VariableElimination step (done with UCVE), the uncertainties
     * are tracked during the cross-sums, which allows pruning actions that are
     * known to be suboptimal.
     */
    class MAUCE {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor needs to know in advance the groups of
             * agents that need to collaboratively cooperate in order to
             * reach their goal. This is converted in a simple Q-Function
             * containing the learned averages for those groups.
             *
             * Note: there can be multiple groups with the same keys (to
             * exploit structure of multiple reward functions between the same
             * agents), but each PartialKeys must be sorted!
             *
             * @param a The factored action space of the problem.
             * @param dependencies The local groups of connected agents.
             * @param ranges The ranges for each local group.
             */
            MAUCE(Action a, const std::vector<PartialKeys> & dependencies, std::vector<double> ranges);

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
            /// The action space
            Action A;
            /// The current timestep, used to compute logtA
            unsigned timestep_;
            /// The averages and counts for the local actions.
            RollingAverage averages_;
            /// The squared ranges for each local group.
            std::vector<double> rangesSquared_;
            /// Precomputed logA since it won't change.
            double logA_;
    };
}

#endif
