#ifndef AI_TOOLBOX_FACTORED_MDP_JOINT_ACTION_LEARNER_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_JOINT_ACTION_LEARNER_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/Factored/MDP/Types.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

#include <AIToolbox/MDP/Algorithms/QLearning.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class represents a single Joint Action Learner agent.
     *
     * A JAL agent learns a QFunction for its own values while keeping track of
     * the actions performed by the other agents with which it is interacting.
     *
     * In order to reason about its own QFunction, a JAL keeps a model of the
     * policies of the other agents. This is done by keeping counters for each
     * actions that other agents have performed, and performing a maximum
     * likelihood computation in order to estimate their policies.
     *
     * While internally a QFunction is kept for the full joint action space,
     * after using the policy models the output will be a normal
     * MDP::QFunction, which can then be used to provide a policy.
     *
     * The internal learning is done using MDP::QLearning.
     *
     * This method does not try to handle factorized states. Here we also
     * assume that the joint action space is of reasonable size, as we allocate
     * an MDP::QFunction for it.
     *
     * \sa AIToolbox::MDP::QLearning
     */
    class JointActionLearner {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param S The size of the state space.
             * @param A The size of the joint action space.
             * @param id The id of this agent in the joint action space.
             * @param discount The discount factor for the QLearning process.
             * @param alpha The learning rate for the QLearning process.
             */
            JointActionLearner(size_t S, Action A, size_t id, double discount = 1.0, double alpha = 0.1);

            /**
             * @brief This function updates the internal joint QFunction.
             *
             * This function updates the counts for the actions of the other
             * agents, and the value of the joint QFunction based on the
             * inputs.
             *
             * Then, it updates the single agent QFunction only for the initial
             * state using the internal counts to update its expected value
             * given the new estimates for the other agents' policies.
             *
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t s, const Action & a, size_t s1, double rew);

            /**
             * @brief This function returns the internal joint QFunction.
             *
             * @return A reference to the internal joint QFunction.
             */
            const AIToolbox::MDP::QFunction & getJointQFunction() const;

            /**
             * @brief This function returns the internal single QFunction.
             *
             * @return A reference to the internal single QFunction.
             */
            const AIToolbox::MDP::QFunction & getSingleQFunction() const;

            /**
             * @brief This function sets the learning rate parameter.
             *
             * The learning parameter determines the speed at which the
             * QFunction is modified with respect to new data. In fully
             * deterministic environments (such as an agent moving through
             * a grid, for example), this parameter can be safely set to
             * 1.0 for maximum learning.
             *
             * The learning rate parameter must be > 0.0 and <= 1.0,
             * otherwise the function will throw an std::invalid_argument.
             *
             * \sa QLearning
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
             * The discount parameter controls the amount that future rewards are considered
             * by QLearning. If 1, then any reward is the same, if obtained now or in a million
             * timesteps. Thus the algorithm will optimize overall reward accretion. When less
             * than 1, rewards obtained in the presents are valued more than future rewards.
             *
             * \sa QLearning
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
             * @brief This function returns the number of states on which JointActionLearner is working.
             *
             * @return The number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the action space on which JointActionLearner is working.
             *
             * @return The action space.
             */
            const Action & getA() const;

            /**
             * @brief This function returns the id of the agent represented by this class.
             *
             * @return The id of this agent.
             */
            size_t getId() const;

        private:
            Action A;
            size_t id_;

            std::vector<unsigned> stateCounters_;
            boost::multi_array<std::vector<unsigned>, 2> stateActionCounts_;

            AIToolbox::MDP::QFunction singleQFun_;
            PartialFactorsEnumerator jointActions_;

            AIToolbox::MDP::QLearning qLearning_;
    };
}

#endif
