#ifndef AI_TOOLBOX_FACTORED_BANDIT_ROLLING_AVERAGE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_BANDIT_ROLLING_AVERAGE_HEADER_FILE

#include <AIToolbox/Factored/Bandit/Types.hpp>

namespace AIToolbox::Factored::Bandit {
    /**
     * @brief This class computes averages and counts for a multi-agent cooperative Bandit problem.
     *
     * This class can be used to compute the averages and counts for all
     * actions in a Bandit problem over time. The class assumes that the
     * problem is factored, and agents depend on each other in smaller groups.
     */
    class RollingAverage {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             * @param dependencies The local groups to record. Multiple groups with the same keys are allowed.
             */
            RollingAverage(Action A, const std::vector<PartialKeys> & dependencies);

            /**
             * @brief This function updates the QFunction and counts.
             *
             * @param a The action taken.
             * @param rews The rewards obtained in the previous timestep, one per agent group.
             */
            void stepUpdateQ(const Action & a, const Rewards & rews);

            /**
             * @brief This function resets the QFunction and counts to zero.
             */
            void reset();

            /**
             * @brief This function returns the size of the action space.
             *
             * @return The size of the action space.
             */
            const Action & getA() const;

            /**
             * @brief This function returns a reference to the internal QFunction.
             *
             * @return A reference to the internal QFunction.
             */
            const QFunction & getQFunction() const;

            /**
             * @brief This function returns a reference for the counts for the actions.
             *
             * @return A reference to the counts of the actions.
             */
            const std::vector<std::vector<unsigned>> & getCounts() const;

            /**
             * @brief This function returns the estimated squared distance of the samples from the mean.
             *
             * The retuned values estimate sum_i (x_i - mean_x)^2 for the
             * rewards of each local action. Note that these values only have
             * meaning when the respective action has at least 2 samples.
             *
             * @return A reference to the estimated square distance from the mean.
             */
            const std::vector<Vector> & getM2s() const;

        private:
            Action A;
            QFunction qfun_;
            std::vector<Vector> M2s_;
            std::vector<std::vector<unsigned>> counts_;
    };
}

#endif
