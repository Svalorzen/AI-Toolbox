#ifndef AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE
#define AI_TOOLBOX_MDP_QLEARNING_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox {
    namespace MDP {

#ifndef DOXYGEN_SKIP
        // This is done to avoid bringing around the enable_if everywhere.
        template <typename M, typename = typename std::enable_if<is_generative_model<M>::value>::type>
        class QLearning;
#endif
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
         * model, and so it is a good algorithm to apply in real world
         * scenarios for example, where there is no way to reproduce the
         * world's behavior aside from actually try out actions. However it is
         * needed to know the size of the state space, the size of the action
         * space and the discount factor of the problem.
         */
        template <typename M>
        class QLearning<M> {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * The learning rate must be > 0.0 and <= 1.0, otherwise the
                 * constructor will throw an std::invalid_argument.
                 *
                 * @param model The MDP model that QLearning will use as a base.
                 * @param alpha The learning rate of the QLearning method.
                 */
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
                 * accordingly. The final behaviour of QLearning is very
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
                 * @brief This function returns a reference to the internal QFunction.
                 *
                 * The returned reference can be used to build Policies, for example
                 * MDP::QGreedyPolicy.
                 *
                 * @return The internal QFunction.
                 */
                const QFunction & getQFunction() const;

                /**
                 * @brief This function returns the MDP generative model being used.
                 *
                 * @return The MDP generative model.
                 */
                const M& getModel() const;

            protected:
                const M & model_;
                // We cache values for max performance.
                size_t S, A;
                double alpha_;
                double discount_;

                QFunction q_;
        };

        template <typename M>
        QLearning<M>::QLearning(const M& model, double alpha) : model_(model), S(model_.getS()), A(model_.getA()), alpha_(alpha), discount_(model_.getDiscount()), q_(makeQFunction(S,A)) {
            if ( alpha_ <= 0.0 || alpha_ > 1.0 )        throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        }

        template <typename M>
        void QLearning<M>::stepUpdateQ(size_t s, size_t a, size_t s1, double rew) {
            q_(s, a) += alpha_ * ( rew + discount_ * q_.row(s1).maxCoeff() - q_(s, a) );
        }

        template <typename M>
        void QLearning<M>::setLearningRate(double a) {
            if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            alpha_ = a;
        }

        template <typename M>
        double QLearning<M>::getLearningRate() const { return alpha_; }

        template <typename M>
        const QFunction & QLearning<M>::getQFunction() const { return q_; }

        template <typename M>
        const M& QLearning<M>::getModel() const { return model_; }
    }
}
#endif
