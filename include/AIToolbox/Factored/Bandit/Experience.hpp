#ifndef AI_TOOLBOX_FACTORED_BANDIT_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_EXPERIENCE_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class computes averages and counts for a multi-agent cooperative Bandit problem.
     *
     * This class can be used to compute the averages and counts for all
     * actions in a Bandit problem over time. The class assumes that the
     * problem is factored, and agents depend on each other in smaller groups.
     */
    class Experience {
        public:
            using VisitsTable = std::vector<std::vector<unsigned long>>;
            using Indeces = std::vector<size_t>;

            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             * @param dependencies The local groups to record. Multiple groups with the same keys are allowed.
             */
            Experience(Action A, const std::vector<PartialKeys> & dependencies);

            /**
             * @brief This function updates the QFunction and counts.
             *
             * This function additionally returns a reference to the indeces
             * updated for each group of agents. This is useful, for example,
             * when updating a model or a policy without needing to recompute
             * these indeces all the time.
             *
             * @param a The action taken.
             * @param rews The rewards obtained in the previous timestep, one per agent group.
             *
             * @return The indeces of each agent group updated.
             */
            const Indeces & record(const Action & a, const Rewards & rews);

            /**
             * @brief This function resets the QFunction and counts to zero.
             */
            void reset();

            /**
             * @brief This function returns the local groups of agents.
             */
            const std::vector<PartialKeys> & getDependencies() const;

            /**
             * @brief This function returns the number of times the record function has been called.
             *
             * @return The number of recorded timesteps.
             */
            unsigned long getTimesteps() const;

            /**
             * @brief This function returns a reference to the internal QFunction.
             *
             * The reward matrix contains the current average rewards computed for each action.
             *
             * @return A reference to the internal QFunction.
             */
            const QFunction & getRewardMatrix() const;

            /**
             * @brief This function returns a reference for the counts for the actions.
             *
             * @return A reference to the counts of the actions.
             */
            const VisitsTable & getVisitsTable() const;

            /**
             * @brief This function returns the estimated squared distance of the samples from the mean.
             *
             * The retuned values estimate sum_i (x_i - mean_x)^2 for the
             * rewards of each local action. Note that these values only have
             * meaning when the respective action has at least 2 samples.
             *
             * @return A reference to the estimated square distance from the mean.
             */
            const std::vector<Vector> & getM2Matrix() const;

            /**
             * @brief This function returns the size of the action space.
             *
             * @return The size of the action space.
             */
            const Action & getA() const;

        private:
            Action A;
            const std::vector<PartialKeys> & deps_;

            QFunction qfun_;
            std::vector<Vector> M2s_;
            VisitsTable counts_;
            Indeces indeces_;

            unsigned long timesteps_;

    };
}

#endif
