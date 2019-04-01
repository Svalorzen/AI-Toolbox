#ifndef AI_TOOLBOX_FACTORED_MDP_SPARSE_COOPERATIVE_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_SPARSE_COOPERATIVE_QLEARNING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>

#include <AIToolbox/Factored/Utils/FilterMap.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class represents the Sparse Cooperative QLearning algorithm.
     *
     * This algorithm is designed to work in cooperative multi-agent
     * problems, but can as easily be used for factored state/action single
     * agent MDPs (since the two things are equivalent).
     *
     * Rather than having a single huge QFunction covering all possible
     * state/action pairs, SparseCooperativeQLearning keeps its QFunction
     * split into QFunctionRule. Each rule covers a specific reward that
     * can be obtained via a PartialState and PartialAction.
     *
     * As the agent interacts with the world, these rules are updated to
     * better reflect the rewards obtained from the environment. At each
     * timestep, each rule applicable on the starting State and Action are
     * updated based on the next State and the optimal Action that is
     * computed with the existing rules via VariableElimination.
     *
     * Aside from this, this algorithm is very similar to the single agent
     * MDP::QLearning (hence the name).
     */
    class SparseCooperativeQLearning {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor initializes all data structures and
             * parameters for the correct functioning of QLearning.
             *
             * Note: This algorithm can be used for bandit problems by
             * simply omitting the state part (giving in an empty vector
             * for states), rather than giving a single state vector. This
             * should speed things up a bit.
             *
             * @param S The factored state space of the environment.
             * @param A The factored action space for the agent.
             * @param discount The discount for future rewards.
             * @param alpha The learning parameter.
             */
            SparseCooperativeQLearning(State S, Action A, double discount, double alpha);

            /**
             * @brief This function reserves memory for at least s rules.
             *
             * @param s The number of rules to be reserved.
             */
            void reserveRules(size_t s);

            /**
             * @brief This function inserts a QFunctionRule in the covered set.
             *
             * @param rule The new rule to cover.
             */
            void insertRule(QFunctionRule rule);

            /**
             * @brief This function returns the number of rules currently stored.
             *
             * @return The number of stored QFunctionRules.
             */
            size_t rulesSize() const;

            /**
             * @brief This function sets the learning rate parameter.
             *
             * The learning parameter determines the speed at which the
             * QFunctions are modified with respect to new data. In fully
             * deterministic environments (such as an agent moving through
             * a grid, for example), this parameter can be safely set to
             * 1.0 for maximum learning.
             *
             * On the other side, in stochastic environments, in order to
             * converge this parameter should be higher when first starting
             * to learn, and decrease slowly over time.
             *
             * Otherwise it can be kept somewhat high if the environment
             * dynamics change progressively, and the algorithm will adapt
             * accordingly. The final behavior of
             * SparseCooperativeQLearning is very dependent on this
             * parameter.
             *
             * The learning rate parameter must be > 0.0 and <= 1.0,
             * otherwise the function will throw an std::invalid_argument.
             *
             * @param a The new learning rate parameter.
             */
            void setLearningRate(double a);

            /**
             * @brief This function will return the current set learning rate parameter.
             *
             * @return The currently set learning rate parameter.
             */
            double getLearningRate() const;

            /**
             * @brief This function sets the new discount parameter.
             *
             * The discount parameter controls the amount that future
             * rewards are considered by SparseCooperativeQLearning. If 1,
             * then any reward is the same, if obtained now or in a million
             * timesteps. Thus the algorithm will optimize overall reward
             * accretion. When less than 1, rewards obtained in the
             * presents are valued more than future rewards.
             *
             * @param d The new discount factor.
             */
            void setDiscount(double d);

            /**
             * @brief This function returns the currently set discount parameter.
             *
             * @return The currently set discount parameter.
             */
            double getDiscount() const;

            /**
             * @brief This function updates the internal QFunctionRules based on experience.
             *
             * This function takes a single experience point and uses it to
             * update the QFunctionRules. Since in order to do this we have
             * to compute the best possible action for the next timestep,
             * we return it in case it is needed.
             *
             * Note: this algorithm expects one reward per factored action
             * (i.e. the size of the action input and the rewards input
             * should be the same)!
             *
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param rew The reward obtained.
             *
             * @return The best action to be performed in the next timestep.
             */
            Action stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & rew);

            /**
             * @brief This function returns the state space on which SparseCooperativeQLearning is working.
             *
             * @return The number of states.
             */
            const State & getS() const;

            /**
             * @brief This function returns the action space on which SparseCooperativeQLearning is working.
             *
             * @return The number of actions.
             */
            const Action & getA() const;

            /**
             * @brief This function returns a reference to the internal FilterMap of QFunctionRules.
             *
             * @return The internal QFunctionRules.
             */
            const FilterMap<QFunctionRule> & getQFunctionRules() const;

        private:
            State S;
            Action A;
            double discount_, alpha_;
            FilterMap<QFunctionRule> rules_;
    };
}

#endif
