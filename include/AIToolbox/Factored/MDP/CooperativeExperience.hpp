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
     *
     * The events are recorded with respect to a given structure, which should
     * match the one of the generative model.
     *
     * Note that since this class contains data in a DDN format, it's probably
     * only usable by directly inspecting the stored VisitTable and
     * RewardMatrix. Thus we don't yet provide general getters for state/action
     * pairs.
     */
    class CooperativeExperience {
        public:
            using RewardMatrix = std::vector<FactoredDDN::Node>;

            // Here we define the visit structure; it's the same as a vector of
            // FactoredDDN::Node, but uses unsigned so we have to use a 2D
            // unsigned table rather than Matrix2D. We also don't really need
            // the tags again, so it's just a vector of vectors.
            using VisitTable = std::vector<std::vector<Table2D>>;

            // Used to avoid recomputation when doing sync in RL.
            using Indeces = std::vector<std::pair<size_t, size_t>>;

            /**
             * @brief Basic constructor.
             *
             * Note that the structure input does not need to pre-allocate the
             * value matrices, nor to fill their values, since we do that
             * internally. Here we only need the structure of the problem.
             *
             * @param s The state space to record.
             * @param a The action space to record.
             * @param structure The coordination graph of the cooperative problem.
             */
            CooperativeExperience(State s, Action a, std::vector<FactoredDDN::Node> structure);

            /**
             * @brief This function adds a new event to the recordings.
             *
             * Note that here we expect a vector of rewards, of the same size
             * as the state space.
             *
             * This function additionally returns a reference to the indeces
             * updated for each element of the underlying DDN. This is useful,
             * for example, when updating the CoordinatedRLModel without
             * needing to recompute these indeces all the time.
             *
             * @param s     Old state.
             * @param a     Performed action.
             * @param s1    New state.
             * @param rew   Obtained rewards.
             *
             * @return The indeces of s and a updated in the DDN.
             */
            const Indeces & record(const State & s, const Action & a, const State & s1, const Rewards & rew);

            /**
             * @brief This function resets all experienced rewards and transitions.
             */
            void reset();

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
            const RewardMatrix & getRewardMatrix() const;

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
            const State & getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            const Action & getA() const;

        private:
            State S;
            Action A;

            VisitTable visits_;
            RewardMatrix rewards_;
            std::vector<std::pair<size_t, size_t>> indeces_;
    };
}

#endif

