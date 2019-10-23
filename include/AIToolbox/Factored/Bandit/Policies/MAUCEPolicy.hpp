#ifndef AI_TOOLBOX_FACTORED_BANDIT_MAUCE_POLICY_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_MAUCE_POLICY_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>
#include <AIToolbox/Factored/Utils/FilterMap.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/UCVE.hpp>
#include <AIToolbox/Factored/Bandit/Experience.hpp>
#include <AIToolbox/Factored/Bandit/Policies/PolicyInterface.hpp>

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
    class MAUCEPolicy : public PolicyInterface {
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
             * @param exp The Experience we learn from.
             * @param ranges The ranges for each local group.
             */
            MAUCEPolicy(const Experience & exp, std::vector<double> ranges);

            /**
             * @brief This function selects an action using MAUCE.
             *
             * We construct an UCVE process, which is able to compute the
             * Action that maximizes the correct overall UCB exploration bonus.
             *
             * UCVE is however a somewhat complex and slow algorithm; for a
             * faster alternative you can look into ThompsonSamplingPolicy.
             *
             * \sa ThompsonSamplingPolicy
             *
             * @return The new optimal action to be taken at the next timestep.
             */
            virtual Action sampleAction() const override;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * As sampleAction() is deterministic, we simply run it to check
             * that the Action it returns is equal to the one passed as input.
             *
             * @param a The selected action.
             *
             * @return This function returns an approximation of the probability of choosing the input action.
             */
            virtual double getActionProbability(const Action & a) const override;

            /**
             * @brief This function returns the RollingAverage learned from the data.
             *
             * These rules skip the exploration part, to allow the creation
             * of a policy using the learned QFunction (since otherwise
             * this algorithm would forever explore).
             *
             * @return The RollingAverage containing all statistics from the input data.
             */
            const Experience & getExperience() const;

        private:
            /// The averages and counts for the local actions.
            const Experience & exp_;
            /// The squared ranges for each local group.
            std::vector<double> rangesSquared_;
            /// Precomputed logA since it won't change.
            double logA_;
    };
}

#endif
