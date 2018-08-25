#ifndef AI_TOOLBOX_MDP_EXPECTED_SARSA_HEADER_FILE
#define AI_TOOLBOX_MDP_EXPECTED_SARSA_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>
#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the ExpectedSARSA algorithm.
     *
     * This algorithm is a subtle improvement over the SARSA algorithm.
     *
     * \sa SARSA
     *
     * The difference between this algorithm and the original SARSA algorithm
     * lies in the value used to approximate the value for the next timestep.
     * In standard SARSA this value is directly taken as the current
     * approximation of the value of the QFunction for the newly sampled state
     * and the next action to be performed (the final "SA" in SAR"SA").
     *
     * In Expected SARSA this value is instead replaced by the expected value
     * for the newly sampled state, given the policy from which we will sample
     * the next action. In this sense Expected SARSA is more similar to
     * QLearning: where QLearning uses the max over the QFunction for the next
     * state, Expected SARSA uses the future expectation over the current
     * online policy.
     *
     * This reduces considerably the variance of the updates performed, which
     * in turn allows to somewhat increase the learning rate for the method,
     * which allows Expected SARSA to learn faster than simple SARSA. All
     * guarantees of normal SARSA are maintained.
     */
    class ExpectedSARSA {
        public:
            /**
             * @brief Basic constructor.
             *
             * Note that differently from normal SARSA, ExpectedSARSA does not
             * self-contain its own QFunction. This is because many policies
             * are implemented in terms of a QFunction continuously updated by
             * a method (e.g. QGreedyPolicy).
             *
             * At the same time ExpectedSARSA needs this policy in order to be
             * able to perform its expected value computation. In order to
             * avoid having a chicken and egg problem, ExpectedSARSA takes a
             * QFunction as parameter to allow the user to create it an use the
             * same one for both ExpectedSARSA and the policy.
             *
             * The learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * @param qfun The QFunction underlying the ExpectedSARSA algorithm.
             * @param policy The policy used to select actions.
             * @param discount The discount of the underlying MDP model.
             * @param alpha The learning rate of the ExpectedSARSA method.
             */
            ExpectedSARSA(QFunction & qfun, const PolicyInterface & policy, double discount = 0.0, double alpha = 0.1);

            /**
             * @brief Basic constructor.
             *
             * Note that differently from normal SARSA, ExpectedSARSA does not
             * self-contain its own QFunction. This is because many policies
             * are implemented in terms of a QFunction continuously updated by
             * a method (e.g. QGreedyPolicy).
             *
             * At the same time ExpectedSARSA needs this policy in order to be
             * able to perform its expected value computation. In order to
             * avoid having a chicken and egg problem, ExpectedSARSA takes a
             * QFunction as parameter to allow the user to create it an use the
             * same one for both ExpectedSARSA and the policy.
             *
             * The learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * This constructor copies the discount parameter from the supplied
             * model. It does not keep the reference, so if the discount needs
             * to change you'll need to update it here manually too.
             *
             * @param qfun The QFunction underlying the ExpectedSARSA algorithm.
             * @param policy The policy used to select actions.
             * @param model The MDP model that ExpectedSARSA will use as a base.
             * @param alpha The learning rate of the ExpectedSARSA method.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            ExpectedSARSA(QFunction & qfun, const PolicyInterface & policy, const M& model, double alpha = 0.1);

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
             * accordingly. The final behaviour of ExpectedSARSA is very
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
             * The discount parameter controls the amount that future rewards
             * are considered by ExpectedSARSA. If 1, then any reward is the
             * same, if obtained now or in a million timesteps. Thus the
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
             * Keep in mind that, since ExpectedSARSA needs to compute the
             * QFunction for the currently used policy, it needs to know two
             * consecutive state-action pairs, in order to correctly relate how
             * the policy acts from state to state.
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
             * @brief This function returns a reference to the policy used by ExpectedSARSA.
             *
             * @return The internal policy reference.
             */
            const PolicyInterface & getPolicy() const;

        private:
            const PolicyInterface & policy_;
            size_t S, A;
            double alpha_;
            double discount_;

            QFunction & q_;
    };

    template <typename M, typename>
    ExpectedSARSA::ExpectedSARSA(QFunction & qfun, const PolicyInterface & policy, const M& model, const double alpha) :
            ExpectedSARSA(qfun, policy, model.getDiscount(), alpha) {}
}
#endif
