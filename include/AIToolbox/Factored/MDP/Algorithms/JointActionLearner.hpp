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
             * NOTE: This function does NOT update the single agent QFunction
             * (which is probably what you want since that is the one that
             * you'll most likely use to produce a policy).
             *
             * This is because the single QFunction update is a very expensive
             * operation, so it is left as a separate operation.
             *
             * This function updates the counts for the actions of the other
             * agents, and the value of the joint QFunction based on the
             * inputs.
             *
             * \sa stepUpdateSingleQ()
             *
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param rew The reward obtained.
             */
            void stepUpdateJointQ(size_t s, const Action & a, size_t s1, double rew);

            /**
             * @brief This function updates the single agent QFunction.
             *
             * This function assumes that you have already updated the joint
             * QFunction via stepUpdateJointQ().
             *
             * This function updates the single agent QFunction only for the
             * specified state (so you most likely want the s1 from the call to
             * stepUpdateJointQ so that you can sample an action from there).
             *
             * The reason this step is separate is that once the model for the
             * other agents has changed (via the counts), the whole single
             * QFunction needs to be recomputed. Needless to say, this is expensive.
             *
             * Since to sample an action only the row with the appropriate
             * state is needed, you can avoid some work by calling this
             * function.
             *
             * @param s1 The state to update (most likely the "next" state s1 from the stepUpdateJointQ call).
             */
            void stepUpdateSingleQ(size_t s1);

            /**
             * @brief This function updates the single agent QFunction.
             *
             * This function assumes that you have already updated the joint
             * QFunction via stepUpdateJointQ().
             *
             * This function updates the WHOLE single agent QFunction. This is
             * an expensive operation, so if the whole update is not needed,
             * please give a look at stepUpdateSingleQ(size_t), which does only
             * a partial update.
             */
            void stepUpdateSingleQ();

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
            Action getA() const;

        private:
            /**
             * @brief This function contains the implementation of the single agent QFunction update.
             *
             * This function updates the single agent QFunction using the joint
             * action QFunction as the source of data. The update is not
             * incremental, but must be performed from scratch as we assume the
             * models for the other agents have changed, changing the values
             * with them.
             *
             * This function takes a range, which in practice is either a
             * single state or all of them.
             *
             * @param begin The beginning of the range of states to update.
             * @param end The end of the range of states to update.
             */
            void updateSingleQFunction(size_t begin, size_t end);

            Action A;
            size_t id_;
            unsigned timestep_;

            std::vector<std::vector<unsigned>> counts_;
            AIToolbox::MDP::QFunction singleQFun_;
            PartialFactorsEnumerator jointActions_;

            AIToolbox::MDP::QLearning qLearning_;
    };
}

#endif
