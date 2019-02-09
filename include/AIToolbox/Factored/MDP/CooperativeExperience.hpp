#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_EXPERIENCE_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_EXPERIENCE_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class keeps track of registered events and rewards.
     *
     * This class is a simple logger of events. It keeps track of both
     * the number of times a particular transition has happened, and the
     * total reward gained in any particular transition. However, it
     * does not record each event separately (i.e. you can't extract
     * the results of a particular transition in the past).
     */
    class CooperativeExperience {
        public:
            using VisitTable = FactoredDDN;
            using RewardTable = FactoredDDN;

            /**
             * @brief Basic constructor.
             *
             * Note that the structure does not need to pre-allocate the value
             * matrices, nor to fill their values, since we do that internally.
             * Here we only need the structure of the problem.
             *
             * @param s The state space to record.
             * @param a The action space to record.
             * @param structure The coordination graph of the cooperative problem.
             */
            CooperativeExperience(State s, Action a, std::vector<FactoredDDN::Node> structure);
            // Experience(State s, Action a, std::vector<FactoredDDN::Node> visits, rewards);

            /**
             * @brief This function adds a new event to the recordings.
             *
             * Note that here we expect a vector of rewards, of the same size
             * as the state space.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             * @param rew   Obtained rewards.
             */
            void record(const State & s, const Action & a, const State & s1, const Rewards & rew);

            /**
             * @brief This function resets all experienced rewards and transitions.
             */
            void reset();

            /**
             * @brief This function returns the current recorded visits for a transitions.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             */
            unsigned long getVisits(size_t s, size_t a, size_t s1) const;

            /**
             * @brief This function returns the number of transitions recorded that start with the specified state and action.
             *
             * @param s     The initial state.
             * @param a     Performed action.
             *
             * @return The total number of transitions that start with the specified state-action pair.
             */
            unsigned long getVisitsSum(size_t s, size_t a) const;

            /**
             * @brief This function returns the cumulative rewards obtained from a specific transition.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             */
            double getReward(const State & s, const Action & a, const State & s1) const;
            double getReward(const PartialState & s, const PartialAction & a, const PartialState & s1) const;

            /**
             * @brief This function returns the total reward obtained from transitions that start with the specified state and action.
             *
             * @param s     The initial state.
             * @param a     Performed action.
             *
             * @return The total number of transitions that start with the specified state-action pair.
             */
            double getRewardSum(size_t s, size_t a) const;

            /**
             * @brief This function returns the visits table for inspection.
             *
             * @return The visits table.
             */
            const VisitTable & getVisitTable() const;

            /**
             * @brief This function returns the rewards table for inspection.
             *
             * @return The rewards table.
             */
            const RewardTable & getRewardTable() const;

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
            State S;
            Action A;

            VisitTable visits_;
            // VisitSumTable visitsSum_;

            RewardTable rewards_;
            // RewardSumTable rewardsSum_;
    };
}

#endif

