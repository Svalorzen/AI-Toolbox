#ifndef AI_TOOLBOX_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_EXPERIENCE_HEADER_FILE

#include <istream>
#include <ostream>

#include <boost/multi_array.hpp>

namespace AIToolbox {
    /**
     * @brief This class keeps track of registered events and rewads.
     *
     * This class is a simple logger of events. It keeps track of both
     * the number of times a particular transition has happened, and the
     * total reward gained in any particular transition. However, it
     * does not record each event separately (i.e. you can't extract
     * the results of a particular transition in the past).
     */
    class Experience {
        public:
            using VisitTable = boost::multi_array<unsigned long,3>;
            using RewardTable = boost::multi_array<double,3>;

            /**
             * @brief Basic constructor.
             *
             * @param S The number of states of the world.
             * @param A The number of actions available to the agent.
             */
            Experience(size_t S, size_t A);

            /**
             * @brief Adds a new event to the recordings.
             *
             * @param s     Old state.
             * @param s1    New state.
             * @param a     Performed action.
             * @param rew   Obtained reward.
             */
            void record(size_t s, size_t s1, size_t a, double rew);

            /**
             * @brief This function resets all experienced rewards and transitions.
             */
            void reset();

            /**
             * @brief This function returns the visits table for inspection.
             *
             * @return The visits table.
             */
            const VisitTable  & getVisits() const;

            /**
             * @brief This function returns the rewards table for inspection.
             *
             * @return The rewards table.
             */
            const RewardTable & getRewards() const;

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

            VisitTable visits_;
            RewardTable rewards_;

            friend std::istream& operator>>(std::istream &os, Experience &);
    };

    std::ostream& operator<<(std::ostream &os, const Experience &);
    std::istream& operator>>(std::istream &is, Experience &);
}

#endif
