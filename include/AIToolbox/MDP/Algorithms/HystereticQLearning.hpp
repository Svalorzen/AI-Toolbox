#ifndef AI_TOOLBOX_MDP_HYSTERETIC_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_HYSTERETIC_QLEARNING_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the Hysteretic QLearning algorithm.
     *
     * This algorithm is a very simple but powerful way to learn the
     * optimal QFunction for an MDP model, where the transition and reward
     * functions are unknown. It works in an offline fashion, meaning that
     * it can be used even if the policy that the agent is currently using
     * is not the optimal one, or is different by the one currently
     * specified by the HystereticQLearning QFunction.
     *
     * \sa QLearning
     *
     * The algorithm functions quite like the normal QLearning algorithm, with
     * a small difference: it has an additional learning parameter, beta.
     *
     * One of the learning parameters (alpha) is used when the change to the
     * underlying QFunction is positive. The other (beta), which should be kept
     * lower than alpha, is used when the change is negative.
     *
     * This is useful when using QLearning for multi-agent RL where each agent
     * is independent. A multi-agent environment is non-stationary from the
     * point of view of a single agent, which is disruptive for normal
     * QLearning and generally prevents it to learn to coordinate with the
     * other agents well.
     *
     * By assigning a higher learning parameter to transitions resulting in a
     * positive feedback, the agent insulates itself from bad results which
     * happen when the other agents take exploratory actions.
     *
     * Bad results are still guaranteed to be discovered, since the learning
     * parameter is still greater than zero, but the algorithm tries to focus
     * on the good things rather than the bad.
     *
     * If the beta parameter is equal to the alpha, this becomes standard
     * QLearning. When the beta parameter is zero, the algorithm becomes
     * equivalent to Distributed QLearning.
     */
    class HystereticQLearning {
        public:
            /**
             * @brief Basic constructor.
             *
             * The alpha learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * The beta learning rate must be >= 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument. It can be zero.
             *
             * Keep in mind that the beta parameter should be lower than the
             * alpha parameter, although this is not enforced.
             *
             * @param S The size of the state space.
             * @param A The size of the action space.
             * @param discount The discount to use when learning.
             * @param alpha The learning rate for positive updates.
             * @param beta The learning rate for negative updates.
             */
            HystereticQLearning(size_t S, size_t A, double discount = 1.0, double alpha = 0.1, double beta = 0.01);

            /**
             * @brief Basic constructor.
             *
             * The alpha learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * The beta learning rate must be >= 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument. It can be zero.
             *
             * Keep in mind that the beta parameter should be lower than the
             * alpha parameter, although this is not enforced.
             *
             * This constructor copies the S and A and discount parameters from
             * the supplied model. It does not keep the reference, so if the
             * discount needs to change you'll need to update it here manually
             * too.
             *
             * @param model The MDP model that HystereticQLearning will use as a base.
             * @param alpha The learning rate of the HystereticQLearning method.
             * @param beta The learning rate for negative updates.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            HystereticQLearning(const M& model, double alpha = 0.1, double beta = 0.01);

            /**
             * @brief This function sets the learning rate parameter for positive updates.
             *
             * The learning parameter determines the speed at which the
             * QFunction is modified with respect to new data, when updates are
             * positive.
             *
             * The learning rate parameter must be > 0.0 and <= 1.0,
             * otherwise the function will throw an std::invalid_argument.
             *
             * @param a The new learning rate parameter for positive updates.
             */
            void setPositiveLearningRate(double a);

            /**
             * @brief This function will return the currently set learning rate parameter for positive updates.
             *
             * @return The currently set learning rate parameter for positive updates.
             */
            double getPositiveLearningRate() const;

            /**
             * @brief This function sets the learning rate parameter for negative updates.
             *
             * The learning parameter determines the speed at which the
             * QFunction is modified with respect to new data, when updates are
             * negative.
             *
             * Note that this parameter can be zero.
             *
             * The learning rate parameter must be >= 0.0 and <= 1.0,
             * otherwise the function will throw an std::invalid_argument.
             *
             * @param b The new learning rate parameter for negative updates.
             */
            void setNegativeLearningRate(double b);

            /**
             * @brief This function will return the currently set learning rate parameter for negative updates.
             *
             * @return The currently set learning rate parameter for negative updates.
             */
            double getNegativeLearningRate() const;

            /**
             * @brief This function sets the new discount parameter.
             *
             * The discount parameter controls the amount that future rewards
             * are considered by HystereticQLearning. If 1, then any reward is
             * the same, if obtained now or in a million timesteps. Thus the
             * algorithm will optimize overall reward accretion. When less than
             * 1, rewards obtained in the presents are valued more than future
             * rewards.
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
             * @brief This function returns the number of states on which HystereticQLearning is working.
             *
             * @return The number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of actions on which HystereticQLearning is working.
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
            double alpha_, beta_;
            double discount_;

            QFunction q_;
    };

    template <typename M, typename>
    HystereticQLearning::HystereticQLearning(const M& model, const double alpha, const double beta) :
            HystereticQLearning(model.getS(), model.getA(), model.getDiscount(), alpha, beta) {}

}
#endif
