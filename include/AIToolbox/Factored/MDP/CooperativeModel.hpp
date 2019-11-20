#ifndef AI_TOOLBOX_FACTORED_COOPERATIVE_MDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_COOPERATIVE_MDP_MODEL_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class models a cooperative MDP.
     *
     * This class can be used in order to model problems where multiple agents
     * cooperate in order to achieve a common goal. In particular, we model
     * problems where each agent only cares about a specific subset of the
     * state space, which allows to build a coordination graph to store
     * dependencies.
     */
    class CooperativeModel {
        public:
            /**
             * @brief Basic constructor
             *
             * @param transitions The transition function.
             * @param rewards The reward function.
             * @param discount The discount factor for the MDP.
             */
            CooperativeModel(DDNGraph graph, DDN::TransitionMatrix transitions, FactoredMatrix2D rewards, double discount = 1.0);

            /**
             * @brief Copy constructor
             *
             * We must manually copy the DDN as it contains a reference to the
             * graph; if this class gets default copied the reference will not
             * point to the internal graph anymore, which will break
             * everything.
             *
             * Note: we copy over the same random state as the next class; this
             * is mostly to copy the behaviour of all other models without an
             * explicit copy constructor. In addition, it makes somewhat easier
             * to reproduce results while moving models around, without
             * worrying whether there are RVO or copies being made.
             *
             * If you want a copy and want to change the random state, just use
             * the other constructor.
             */
            CooperativeModel(const CooperativeModel &);

            /**
             * @brief This function samples the MDP with the specified state action pair.
             *
             * This function samples the model for simulated experience.
             * The transition and reward functions are used to produce,
             * from the state action pair inserted as arguments, a possible
             * new state with respective reward. The new state is picked
             * from all possible states that the MDP allows transitioning
             * to, each with probability equal to the same probability of
             * the transition in the model. After a new state is picked,
             * the reward is the corresponding reward contained in the
             * reward function.
             *
             * @param s The state that needs to be sampled.
             * @param a The action that needs to be sampled.
             *
             * @return A tuple containing a new state and a reward.
             */
            std::tuple<State, double> sampleSR(const State & s, const Action & a) const;

            /**
             * @brief This function samples the MDP with the specified state action pair.
             *
             * This function is equivalent to sampleSR(const State &, const Action &).
             *
             * The only difference is that it allows to output the new State
             * into a pre-allocated State, avoiding the need for an allocation
             * at every sample.
             *
             * NO CHECKS for nullptr are done.
             *
             * @param s The state that needs to be sampled.
             * @param a The action that needs to be sampled.
             * @param s1 The new state.
             *
             * @return The reward for the sampled transition.
             */
            double sampleSR(const State & s, const Action & a, State * s1) const;

            /**
             * @brief This function samples the MDP with the specified state action pair.
             *
             * This function samples the model for simulate experience. The transition
             * and reward functions are used to produce, from the state action pair
             * inserted as arguments, a possible new state with respective reward.
             * The new state is picked from all possible states that the MDP allows
             * transitioning to, each with probability equal to the same probability
             * of the transition in the model.
             *
             * After a new state is picked, the reward is the vector of
             * corresponding rewards contained in the reward function. This
             * means that the vector will have a length equal to the number of
             * bases of the reward function.
             *
             * @param s The state that needs to be sampled.
             * @param a The action that needs to be sampled.
             *
             * @return A tuple containing a new state and a reward.
             */
            std::tuple<State, Rewards> sampleSRs(const State & s, const Action & a) const;

            /**
             * @brief This function samples the MDP with the specified state action pair.
             *
             * This function is equivalent to sampleSRs(const State &, const Action &).
             *
             * The only difference is that it allows to output the new State
             * and Rewards into a pre-allocated State and Rewards, avoiding the
             * need for an allocation at every sample.
             *
             * NO CHECKS for nullptr are done.
             *
             * @param s The state that needs to be sampled.
             * @param a The action that needs to be sampled.
             * @param s1 The new state.
             * @param rews The new rewards.
             */
            void sampleSRs(const State & s, const Action & a, State * s1, Rewards * rews) const;

            /**
             * @brief This function sets a new discount factor for the Model.
             *
             * @param d The new discount factor for the Model.
             */
            void setDiscount(double d);

            /**
             * @brief This function returns the state space of the world.
             *
             * @return The state space.
             */
            const State & getS() const;

            /**
             * @brief This function returns the action space of the MDP.
             *
             * @return The action space.
             */
            const Action & getA() const;

            /**
             * @brief This function returns the currently set discount factor.
             *
             * @return The currently set discount factor.
             */
            double getDiscount() const;

            /**
             * @brief This function returns the stored transition probability for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The probability of the specified transition.
             */
            double getTransitionProbability(const State & s, const Action & a, const State & s1) const;

            /**
             * @brief This function returns the stored expected reward for the specified transition.
             *
             * @param s The initial state of the transition.
             * @param a The action performed in the transition.
             * @param s1 The final state of the transition.
             *
             * @return The expected reward of the specified transition.
             */
            double getExpectedReward(const State & s, const Action & a, const State & s1) const;

            /**
             * @brief This function returns the transition function of the MDP.
             *
             * @return The transition function of the MDP.
             */
            const DDN & getTransitionFunction() const;

            /**
             * @brief This function returns the reward function of the MDP.
             *
             * @return The reward function of the MDP.
             */
            const FactoredMatrix2D & getRewardFunction() const;

            /**
             * @brief This function returns the underlying DDNGraph of the CooperativeExperience.
             *
             * @return The underlying DDNGraph.
             */
            const DDNGraph & getGraph() const;

        private:
            double discount_;

            DDNGraph graph_;
            DDN transitions_;
            FactoredMatrix2D rewards_;

            mutable RandomEngine rand_;
    };
}

#endif
