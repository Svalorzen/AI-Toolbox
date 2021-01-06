#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_QLEARNING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>

#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class represents the Cooperative QLearning algorithm.
     *
     * This is the same as SparseCooperativeQLearning, but we handle dense
     * factored spaces. This obviously is less flexible, but is computationally
     * much faster and can help scale SCQL to larger problems.
     */
    class CooperativeQLearning {
        public:
            /**
             * @brief Basic constructor.
             *
             * This constructor initializes all data structures and
             * parameters for the correct functioning of QLearning.
             *
             * The Q-function is constructed so that each factor has a domain
             * equal to the DDN parents of the relative input basisDomain.
             *
             * @param g The DDN of the environment.
             * @param basisDomains The domains of the Q-Function to use.
             * @param discount The discount for future rewards.
             * @param alpha The learning parameter.
             */
            // SparseCooperativeQLearning(State S, Action A, double discount, double alpha);
            CooperativeQLearning(const DDNGraph & g, const std::vector<std::vector<size_t>> & basisDomains, double discount, double alpha);

            /**
             * @brief This function updates the internal QFunction based on experience.
             *
             * This function takes a single experience point and uses it to
             * update the QFunction. Since in order to do this we have to
             * compute the best possible action for the next timestep, we
             * return it in case it is needed.
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
             * @brief This function returns the DDN on which SparseCooperativeQLearning is working.
             *
             * @return The number of states.
             */
            const DDNGraph & getGraph() const;

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
             * @brief This function returns a reference to the internal QFunction.
             *
             * @return The internal QFunction.
             */
            const FactoredMatrix2D & getQFunction() const;

            /**
             * @brief This function sets the QFunction to a set value.
             *
             * This function is useful to perform optimistic initialization.
             *
             * @param val The value to set all entries in the QFunction.
             */
            void setQFunction(double val);

        private:
            const DDNGraph & graph_;
            double discount_, alpha_;
            FactoredMatrix2D q_;
            QGreedyPolicy policy_;
            // Helper
            Vector agentNormRews_;
    };
}

#endif
