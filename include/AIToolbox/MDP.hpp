#ifndef AI_TOOLBOX_MDP_HEADER_FILE
#define AI_TOOLBOX_MDP_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Experience.hpp>

namespace AIToolbox {
    /**
     * @brief This class models \c Experience as a Markov Decision Process.
     *
     * This class normalizes an \c Experience object to produce a transition function 
     * and a reward function. The transition function is guaranteed to be a correct
     * probability function, as in the sum of the probabilities of all transitions
     * from a particular state and a particular action is always 1.
     * The instance allows modification of the underlying \c Experience object, and is
     * not directly synced with it. This is to avoid possible overheads, as the user
     * can optimize better syncs depending on their use case. See update().
     */
    class MDP {
        public:
            using TransitionTable   = Table3D;
            using RewardTable       = Table3D;

            /**
             * @brief Simple constructor with no Experience.
             * 
             * This constructor guarantees that after the MDP has been
             * built the transition and reward functions are synced to
             * the underlying Experience.
             *
             * @param S The number of states of the world.
             * @param A The number of actions available to the agent.
             */
            MDP(size_t S, size_t A);

            /**
             * @brief Constructor using previous Experience.
             * 
             * This constructor guarantees that after the MDP has been
             * built the transition and reward functions are synced to
             * the underlying Experience.
             *
             * @param exp The base Experience of the model.
             */
            MDP(Experience exp);

            /**
             * @brief This function syncs the MDP to the underlying Experience.
             * 
             * Since use cases in AI are very varied, one may not want to update
             * its MDP for each single transition experienced by the agent. To
             * avoid this we leave to the user the task of syncing between the
             * underlying Experience and the MDP, as he/she sees fit.
             *
             * After this function is run the transition and reward functions
             * will accurately reflect the state of the underlying Experience.
             */
            void                        update();

            /**
             * @brief This function syncs a state action pair in the MDP to the underlying Experience.
             *
             * Since use cases in AI are very varied, one may not want to update
             * its MDP for each single transition experienced by the agent. To
             * avoid this we leave to the user the task of syncing between the
             * underlying Experience and the MDP, as he/she sees fit.
             *
             * This function updates a single state action pair with the underlying
             * Experience. This function is offered to avoid having to recompute the
             * whole MDP if the user knows that only few transitions have been 
             * experienced by the agent.
             *
             * After this function is run the transition and reward functions
             * will accurately reflect the state of the underlying Experience
             * for the specified state action pair.
             *
             * @param s The state that needs to be synced.
             * @param a The action that needs to be synced.
             */
            void                        update(size_t s, size_t a);

            /**
             * @brief This function samples the MDP for the specified state action pair.
             *
             * This function samples the model for simulate experience. The transition
             * and reward functions are used to produce, from the state action pair
             * inserted as arguments, a possible new state with respective reward.
             * The new state is picked from all possible states that the MDP allows
             * transitioning to, each with probability equal to the same probability 
             * of the transition in the model. After a new state is picked, the reward
             * is the corresponding reward contained in the reward function.
             *
             * @param s The state that needs to be sampled.
             * @param a The action that needs to be sampled.
             *
             * @return A tuple containing a new state and a reward.
             */
            std::tuple<size_t, double>  sample(size_t s, size_t a) const;

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

            /**
             * @brief This functions enables access to the underlying Experience.
             * 
             * This function allows the user to modify and insert new data into the
             * underlying Experience of the MDP. Should the Experience be modified,
             * the user will have to manually update the MDP (see update()) so
             * that the new Experience and the MDP are in sync.
             *
             * @return The underlying Experience of the MDP.
             */
            Experience & getExperience();

            /**
             * @brief This function enables viewing of the underlying Experience for const MDPs.
             *
             * @return The underlying Experience of the MDP.
             */
            const Experience & getExperience() const;

            /**
             * @brief This function returns the transition table for inspection.
             *
             * @return The rewards table.
             */
            const TransitionTable & getTransitionFunction() const;

            /**
             * @brief This function returns the rewards table for inspection.
             *
             * @return The rewards table.
             */
            const RewardTable &     getRewardFunction()     const;
        private:
            size_t S, A;

            Experience experience_;

            TransitionTable transitions_;
            RewardTable rewards_;

            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };
}

#endif
