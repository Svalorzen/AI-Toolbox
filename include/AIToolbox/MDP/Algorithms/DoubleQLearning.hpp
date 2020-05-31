#ifndef AI_TOOLBOX_MDP_DOUBLE_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_DOUBLE_QLEARNING_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the double QLearning algorithm.
     *
     * The QLearning algorithm is biased to overestimate the expected future
     * reward during the Bellman equation update, as the bootstrapped max over
     * the same QFunction is actually an unbiased estimator for the expected
     * max, rather than the max expected.
     *
     * This is a problem for certain classes of problems, and DoubleQLearning
     * tries to fix that.
     *
     * DoubleQLearning maintains two separate QFunctions, and in a given
     * timestep one is selected randomly to be updated. The update has the same
     * form as the standard QLearning update, except that the *other* QFunction
     * is used to estimate the expected future reward. The math shows that this
     * technique still results in a bias estimation, but in this case we tend
     * to underestimate.
     *
     * We can still try to counteract this with optimistic initialization, and
     * the final result is often more stable than simple QLearning.
     *
     * Since action selection should be performed w.r.t. both QFunctions,
     * DoubleQLearning stores two things: the first QFunction, and the sum
     * between the first QFunction and the second. The second QFunction is not
     * stored explicitly, and is instead always computed on-the-fly when
     * needed.
     *
     * We do this so we can easily return the sum of both QFunction to apply a
     * Policy to, without the need to store three separate QFunctions
     * explicitly (lowering a bit the memory requirements).
     *
     * If you are interested in the actual values stored in the two "main"
     * QFunctions, please use getQFunctionA() and getQFunctionB(). Note that
     * getQFunctionB() will not return a reference!
     */
    class DoubleQLearning {
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
             * @param alpha The learning rate of the DoubleQLearning method.
             */
            DoubleQLearning(size_t S, size_t A, double discount = 1.0, double alpha = 0.1);

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
             * @param model The MDP model that DoubleQLearning will use as a base.
             * @param alpha The learning rate of the DoubleQLearning method.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            DoubleQLearning(const M& model, double alpha = 0.1);

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
             * accordingly. The final behavior of DoubleQLearning is very
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
             * are considered by DoubleQLearning. If 1, then any reward is the
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
             * @param s The previous state.
             * @param a The action performed.
             * @param s1 The new state.
             * @param rew The reward obtained.
             */
            void stepUpdateQ(size_t s, size_t a, size_t s1, double rew);

            /**
             * @brief This function returns the number of states on which DoubleQLearning is working.
             *
             * @return The number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of actions on which DoubleQLearning is working.
             *
             * @return The number of actions.
             */
            size_t getA() const;

            /**
             * @brief This function returns a reference to the internal "sum" QFunction.
             *
             * The QFunction that is returned does not contain "true" values,
             * but instead is the sum of the two QFunctions that are being
             * updated by DoubleQLearning. This is to make it possible to
             * select actions using standard policy classes.
             *
             * The returned reference can be used to build Policies, for example
             * MDP::QGreedyPolicy.
             *
             * @return The internal "sum" QFunction.
             */
            const QFunction & getQFunction() const;

            /**
             * @brief This function returns a reference to the first internal QFunction.
             *
             * The returned reference can be used to build Policies, for
             * example MDP::QGreedyPolicy, but you should probably use
             * getQFunction() for that.
             *
             * @return The internal first QFunction.
             */
            const QFunction & getQFunctionA() const;

            /**
             * @brief This function returns a copy to the second QFunction.
             *
             * This QFunction is constructed on the fly, and so is not returned by reference!
             *
             * @return What the second QFunction should be.
             */
            QFunction getQFunctionB() const;

            /**
             * @brief This function allows to directly set the internal QFunctions.
             *
             * This can be useful in order to use a QFunction that has already
             * been computed elsewhere. DoubleQLearning will then continue
             * building upon it.
             *
             * Both first and second interal QFunctions are set to the input,
             * while the "sum" QFunction is set to double the input.
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

            mutable RandomEngine rand_;
            std::bernoulli_distribution dist_;

            // First QFunction and "sum" QFunction
            QFunction qa_, qc_;
    };

    template <typename M, typename>
    DoubleQLearning::DoubleQLearning(const M& model, const double alpha) :
            DoubleQLearning(model.getS(), model.getA(), model.getDiscount(), alpha) {}
}

#endif
