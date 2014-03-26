#ifndef AI_TOOLBOX_MDP_RLMODEL_HEADER_FILE
#define AI_TOOLBOX_MDP_RLMODEL_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/MDP/Experience.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class models Experience as a Markov Decision Process.
         *
         * This class normalizes an Experience object to produce a transition function
         * and a reward function. The transition function is guaranteed to be a correct
         * probability function, as in the sum of the probabilities of all transitions
         * from a particular state and a particular action is always 1.
         * Each instance is not directly synced with the supplied Experience object.
         * This is to avoid possible overheads, as the user can optimize better
         * depending on their use case. See sync().
         */
        class RLModel {
            public:
                using TransitionTable   = Table3D;
                using RewardTable       = Table3D;
                /**
                 * @brief Constructor using previous Experience.
                 *
                 * This constructor selects the Experience that will
                 * be used to learn an MDP Model from the data, and initializes
                 * internal Model data.
                 *
                 * The user can choose whether he wants to directly sync
                 * the RLModel to the underlying Experience, or delay
                 * it for later.
                 *
                 * In the latter case the default transition function
                 * defines a transition of probability 1 for each
                 * state to itself, no matter the action.
                 * 
                 * In general it would be better to add some amount of bias
                 * to the Experience so that when a new state-action pair is
                 * tried, the RLModel doesn't automatically compute 100%
                 * probability of transitioning to the resulting state, but
                 * smooths into it. This may depend on your problem though.
                 *
                 * The default reward function is 0.
                 *
                 * @param exp The base Experience of the model.
                 */
                RLModel(const Experience & exp, bool sync);

                /**
                 * @brief This function syncs the whole RLModel to the underlying Experience.
                 *
                 * Since use cases in AI are very varied, one may not want to update
                 * its RLModel for each single transition experienced by the agent. To
                 * avoid this we leave to the user the task of syncing between the
                 * underlying Experience and the RLModel, as he/she sees fit.
                 *
                 * After this function is run the transition and reward functions
                 * will accurately reflect the state of the underlying Experience.
                 */
                void sync();

                /**
                 * @brief This function syncs a state action pair in the RLModel to the underlying Experience.
                 *
                 * Since use cases in AI are very varied, one may not want to update
                 * its RLModel for each single transition experienced by the agent. To
                 * avoid this we leave to the user the task of syncing between the
                 * underlying Experience and the RLModel, as he/she sees fit.
                 *
                 * This function updates a single state action pair with the underlying
                 * Experience. This function is offered to avoid having to recompute the
                 * whole RLModel if the user knows that only few transitions have been
                 * experienced by the agent.
                 *
                 * After this function is run the transition and reward functions
                 * will accurately reflect the state of the underlying Experience
                 * for the specified state action pair.
                 *
                 * @param s The state that needs to be synced.
                 * @param a The action that needs to be synced.
                 */
                void sync(size_t s, size_t a);

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
                std::pair<size_t, double> sample(size_t s, size_t a) const;

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
                 * @brief This function enables inspection of the underlying Experience of the RLModel.
                 *
                 * @return The underlying Experience of the RLModel.
                 */
                const Experience & getExperience() const;

                /**
                 * @brief This function returns the stored transition probability for the specified transition.
                 *
                 * @param s The initial state of the transition.
                 * @param s1 The final state of the transition.
                 * @param a The action performed in the transition.
                 *
                 * @return The probability of the specified transition.
                 */
                double getTransitionProbability(size_t s, size_t s1, size_t a) const;

                /**
                 * @brief This function returns the stored expected reward for the specified transition.
                 *
                 * @param s The initial state of the transition.
                 * @param s1 The final state of the transition.
                 * @param a The action performed in the transition.
                 *
                 * @return The expected reward of the specified transition.
                 */
                double getExpectedReward(size_t s, size_t s1, size_t a) const;

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

                const Experience & experience_;

                TransitionTable transitions_;
                RewardTable rewards_;

                mutable std::default_random_engine rand_;
                mutable std::uniform_real_distribution<double> sampleDistribution_;
        };
    }
}

#endif
