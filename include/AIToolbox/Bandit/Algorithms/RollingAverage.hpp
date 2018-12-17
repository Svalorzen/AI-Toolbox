#ifndef AI_TOOLBOX_BANDIT_ROLLING_AVERAGE_HEADER_FILE
#define AI_TOOLBOX_BANDIT_ROLLING_AVERAGE_HEADER_FILE

#include <AIToolbox/Bandit/Types.hpp>

namespace AIToolbox::Bandit {
    /**
     * @brief This class computes averages and counts for a Bandit problem.
     *
     * This class can be used to compute the averages and counts for all
     * actions in a Bandit problem over time.
     */
    class RollingAverage {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param A The size of the action space.
             */
            RollingAverage(size_t A);

            /**
             * @brief This function updates the QFunction and counts.
             *
             * @param a The action taken.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t a, double rew);

            /**
             * @brief This function returns the size of the action space.
             *
             * @return The size of the action space.
             */
            size_t getA() const;

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
            const std::vector<unsigned> & getCounts() const;

        private:
            QFunction q_;
            std::vector<unsigned> counts_;
    };
}

#endif
