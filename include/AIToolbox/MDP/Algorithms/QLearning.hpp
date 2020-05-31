#ifndef AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the QLearning algorithm.
     *
     * This algorithm is a very simple but powerful way to learn the
     * optimal QFunction for an MDP model, where the transition and reward
     * functions are unknown. It works in an offline fashion, meaning that
     * it can be used even if the policy that the agent is currently using
     * is not the optimal one, or is different by the one currently
     * specified by the QLearning QFunction.
     *
     * The idea is to progressively update the QFunction averaging all
     * obtained datapoints. This can be done by generating data via the
     * model, or by simply sending the agent into the world to try stuff
     * out. This allows to avoid modeling directly the transition and
     * reward functions for unknown problems.
     *
     * This algorithm is guaranteed convergence for stationary MDPs (MDPs
     * that do not change their transition and reward functions over time),
     * given that the learning parameter converges to 0 over time.
     *
     * \sa setLearningRate(double)
     *
     * At the same time, this algorithm can be used for non-stationary
     * MDPs, and it will try to constantly keep up with changes in the
     * environment, given that they are not huge.
     *
     * This algorithm does not actually need to sample from the input
     * model, and so it can be a good algorithm to apply in real world
     * scenarios, where there would be no way to reproduce the world's
     * behavior aside from actually trying out actions. However it is
     * needed to know the size of the state space, the size of the action
     * space and the discount factor of the problem.
     */
    class QLearning {
        public:
            /**
             * @brief Basic constructor.
             *
             * The learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * @param S The size of the state space.
             * @param A The size of the action space.
             * @param discount The discount to use when learning.
             * @param alpha The learning rate of the QLearning method.
             */
            QLearning(size_t S, size_t A, double discount = 1.0, double alpha = 0.1);

            /**
             * @brief Basic constructor.
             *
             * The learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * This constructor copies the S and A and discount parameters from
             * the supplied model. It does not keep the reference, so if the
             * discount needs to change you'll need to update it here manually
             * too.
             *
             * @param model The MDP model that QLearning will use as a base.
             * @param alpha The learning rate of the QLearning method.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            QLearning(const M& model, double alpha = 0.1);

            /**
             * @brief This function sets the learning rate parameter.
             *
             * The learning parameter determines the speed at which the
             * QFunction is modified with respect to new data. In fully
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
             * accordingly. The final behavior of QLearning is very
             * dependent on this parameter.
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
             * The discount parameter controls the amount that future rewards are considered
             * by QLearning. If 1, then any reward is the same, if obtained now or in a million
             * timesteps. Thus the algorithm will optimize overall reward accretion. When less
             * than 1, rewards obtained in the presents are valued more than future rewards.
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
             * @brief This function updates the internal QFunction using the discount set during construction.
             *
             * This function takes a single experience point and uses it to
             * update the QFunction. This is a very efficient method to
             * keep the QFunction up to date with the latest experience.
             *
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t s, size_t a, size_t s1, double rew);

            /**
             * @brief This function returns the number of states on which QLearning is working.
             *
             * @return The number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of actions on which QLearning is working.
             *
             * @return The number of actions.
             */
            size_t getA() const;

            /**
             * @brief This function returns a reference to the internal QFunction.
             *
             * The returned reference can be used to build Policies, for example
             * MDP::QGreedyPolicy.
             *
             * @return The internal QFunction.
             */
            const QFunction & getQFunction() const;

            /**
             * @brief This function allows to directly set the internal QFunction.
             *
             * This can be useful in order to use a QFunction that has already
             * been computed elsewhere. QLearning will then continue building
             * upon it.
             *
             * This is used for example in the Dyna2 algorithm.
             *
             * @param qfun The new QFunction to set.
             */
            void setQFunction(const QFunction & qfun);

        private:
            size_t S, A;
            double alpha_;
            double discount_;

            QFunction q_;
    };

    template <typename M, typename>
    QLearning::QLearning(const M& model, const double alpha) :
            QLearning(model.getS(), model.getA(), model.getDiscount(), alpha) {}
}
#endif
