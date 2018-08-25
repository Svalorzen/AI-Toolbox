#ifndef AI_TOOLBOX_MDP_SARSA_HEADER_FILE
#define AI_TOOLBOX_MDP_SARSA_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the SARSA algorithm.
     *
     * This algorithm is a very simple but powerful way to learn a
     * QFunction for an MDP model, where the transition and reward
     * functions are unknown. It works in an online fashion, meaning that
     * the QFunction learned is the one of the currently used policy.
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
     * The main difference between this algorithm and QLearning is that
     * QLearning always tries to learn the optimal policy, regardless of
     * the one that is currently being executed. Instead, SARSA tries to
     * find a policy which can perform decently given exploration tradeoffs
     * that must be done when learning the QFunction of a new environment.
     * A possible use for this would be to run SARSA together with
     * QLearning; during the training phase one would use SARSA actions in
     * order to perform decently during the training. Afterwards, one could
     * switch to the optimal policy learnt offline by QLearning.
     *
     * This algorithm does not actually need to sample from the input
     * model, and so it can be a good algorithm to apply in real world
     * scenarios, where there would be no way to reproduce the world's
     * behavior aside from actually trying out actions. However it is
     * needed to know the size of the state space, the size of the action
     * space and the discount factor of the problem.
     */
    class SARSA {
        public:
            /**
             * @brief Basic constructor.
             *
             * The learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * @param S The state space of the underlying model.
             * @param A The action space of the underlying model.
             * @param discount The discount of the underlying model.
             * @param alpha The learning rate of the SARSA method.
             */
            SARSA(size_t S, size_t A, double discount = 1.0, double alpha = 0.1);

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
             * @param model The MDP model that SARSA will use as a base.
             * @param alpha The learning rate of the SARSA method.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            SARSA(const M& model, double alpha = 0.1);

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
             * accordingly. The final behaviour of SARSA is very
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
             * by SARSA. If 1, then any reward is the same, if obtained now or in a million
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
             * Keep in mind that, since SARSA needs to compute the
             * QFunction for the currently used policy, it needs to know
             * two consecutive state-action pairs, in order to correctly
             * relate how the policy acts from state to state.
             *
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param a1 The action performed in the new state.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t s, size_t a, size_t s1, size_t a1, double rew);

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

        private:
            size_t S, A;
            double alpha_;
            double discount_;

            QFunction q_;
    };

    template <typename M, typename>
    SARSA::SARSA(const M& model, const double alpha) :
            SARSA(model.getS(), model.getA(), model.getDiscount(), alpha) {}
}
#endif
