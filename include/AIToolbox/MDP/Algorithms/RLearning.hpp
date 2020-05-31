#ifndef AI_TOOLBOX_MDP_RLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_RLEARNING_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the RLearning algorithm.
     *
     * This algorithm is an analogue to QLearning, when one wishes to learn to
     * maximize average reward in infinitely long episodes, rather than
     * discounted reward. Such policies are called T-optimal policies.
     *
     * Indeed, RLearning makes the point that discount is an unnecessary and
     * harmful abstraction in these cases, and that it is generally only used
     * to bound the expected reward when acting infinitely. At the same time,
     * discounting can result in policies which are unnecessarily greedy and
     * don't maximize average reward over time.
     *
     * Thus, the update rule for the QFunction is slightly altered, so that,
     * for each state-action pair, we learn the expected *average-adjusted*
     * reward (present and future), i.e.  the reward minus the average reward,
     * which is the measure we want to learn to act upon. To do so, we also
     * need to learn the average reward.
     *
     * The two elements are learned side by side, and this is why here we have
     * two separate learning rates; one for the QFunction and the other for the
     * average reward. Note that the original paper calls these respectively
     * the beta and alpha learning rate. Here, to keep consistency between
     * methods, we call these alpha and rho. We also rename the standard
     * setLearningRate() function to make sure that users understand what they
     * are setting.
     *
     * \sa setAlphaLearningRate(double)
     * \sa setRhoLearningRate(double)
     *
     * This algorithm does not actually need to sample from the input
     * model, and so it can be a good algorithm to apply in real world
     * scenarios, where there would be no way to reproduce the world's
     * behavior aside from actually trying out actions. However it is
     * needed to know the size of the state space, the size of the action
     * space and the discount factor of the problem.
     */
    class RLearning {
        public:
            /**
             * @brief Basic constructor.
             *
             * Both learning rates must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * @param S The size of the state space.
             * @param A The size of the action space.
             * @param alpha The learning rate for the QFunction.
             * @param rho The learning rate for the average reward.
             */
            RLearning(size_t S, size_t A, double alpha = 0.1, double rho = 0.1);

            /**
             * @brief Basic constructor.
             *
             * Both learning rates must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * This constructor copies the S and A and discount parameters from
             * the supplied model. It does not keep the reference, so if the
             * discount needs to change you'll need to update it here manually
             * too.
             *
             * @param model The MDP model that QLearning will use as a base.
             * @param alpha The learning rate for the QFunction.
             * @param rho The learning rate for the average reward.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            RLearning(const M& model, double alpha = 0.1, double rho = 0.1);

            /**
             * @brief This function sets the learning rate parameter for the QFunction.
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
             * @param a The new alpha learning rate parameter.
             */
            void setAlphaLearningRate(double a);

            /**
             * @brief This function will return the current set alpha learning rate parameter.
             *
             * @return The currently set alpha learning rate parameter.
             */
            double getAlphaLearningRate() const;

            /**
             * @brief This function sets the learning rate parameter for the average reward.
             *
             * The learning parameter determines the speed at which the
             * average reward is modified with respect to new data.
             *
             * The learning rate parameter must be > 0.0 and <= 1.0,
             * otherwise the function will throw an std::invalid_argument.
             *
             * @param r The new rho learning rate parameter.
             */
            void setRhoLearningRate(double r);

            /**
             * @brief This function will return the current set rho learning rate parameter.
             *
             * @return The currently set rho learning rate parameter.
             */
            double getRhoLearningRate() const;

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
             * @brief This function returns the learned average reward.
             *
             * @return The learned average reward.
             */
            double getAverageReward() const;

            /**
             * @brief This function allows to directly set the internal QFunction.
             *
             * This can be useful in order to use a QFunction that has already
             * been computed elsewhere. RLearning will then continue building
             * upon it.
             *
             * This is used for example in the Dyna2 algorithm.
             *
             * @param qfun The new QFunction to set.
             */
            void setQFunction(const QFunction & qfun);

        private:
            size_t S, A;
            double alpha_, rho_;
            double rAvg_;

            QFunction q_;
    };

    template <typename M, typename>
    RLearning::RLearning(const M& model, const double alpha, const double rho) :
            RLearning(model.getS(), model.getA(), alpha, rho) {}
}
#endif
