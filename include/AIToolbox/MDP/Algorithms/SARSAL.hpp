#ifndef AI_TOOLBOX_MDP_SARSAL_HEADER_FILE
#define AI_TOOLBOX_MDP_SARSAL_HEADER_FILE

#include <stddef.h>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Utils.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class represents the SARSAL algorithm.
     *
     * This algorithms adds eligibility traces to the SARSA algorithm.
     *
     * \sa SARSA
     *
     * In order to more effectively use the data obtained, SARSAL keeps a list
     * of previously visited state/action pairs, which are updated together
     * with the last experienced transition. The updates all use the same
     * value, with the difference that state/action pairs experienced more in
     * the past are updated less (by discount*lambda per each previous
     * timestep). Once this reducing coefficient falls below a certain
     * threshold, the old state/action pair is forgotten and not updated
     * anymore. If instead the pair is visited again, the coefficient is once
     * again increased.
     *
     * The idea is to be able to give credit to past actions for current reward
     * in an efficient manner. This reduces the amount of data needed in order
     * to backpropagate rewards, and allows SARSAL to learn faster.
     *
     * This particular version of the algorithm implements capped traces: every
     * time an action/state pair is witnessed, its eligibility trace is reset
     * to 1.0. This avoids potentially diverging values which can happen with
     * the normal eligibility traces.
     */
    class SARSAL {
        public:
            using Trace = std::tuple<size_t, size_t, double>;
            using Traces = std::vector<Trace>;
            /**
             * @brief Basic constructor.
             *
             * The learning rate must be > 0.0 and <= 1.0, otherwise the
             * constructor will throw an std::invalid_argument.
             *
             * @param S The state space of the underlying model.
             * @param A The action space of the underlying model.
             * @param discount The discount of the underlying model.
             * @param alpha The learning rate of the SARSAL method.
             * @param lambda The lambda parameter for the eligibility traces.
             * @param tolerance The cutoff point for eligibility traces.
             */
            SARSAL(size_t S, size_t A, double discount = 1.0, double alpha = 0.1, double lambda = 0.9, double tolerance = 0.001);

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
             * @param model The MDP model that SARSAL will use as a base.
             * @param alpha The learning rate of the SARSAL method.
             * @param lambda The lambda parameter for the eligibility traces.
             * @param tolerance The cutoff point for eligibility traces.
             */
            template <typename M, typename = std::enable_if_t<is_generative_model_v<M>>>
            SARSAL(const M& model, double alpha = 0.1, double lambda = 0.9, double tolerance = 0.001);

            /**
             * @brief This function updates the internal QFunction using the discount set during construction.
             *
             * This function takes a single experience point and uses it to
             * update the QFunction. This is a very efficient method to
             * keep the QFunction up to date with the latest experience.
             *
             * Keep in mind that, since SARSAL needs to compute the
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
             * accordingly. The final behaviour of SARSAL is very
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
             * by SARSAL. If 1, then any reward is the same, if obtained now or in a million
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
             * @brief This function sets the new lambda parameter.
             *
             * This parameter determines how much to decrease updates for each
             * timestep in the past. If set to zero, SARSAL effectively becomes
             * equivalent to SARSA, as no backpropagation will be performed. If
             * set to 1 it will result in a method similar to Monte Carlo
             * sampling, where rewards are backed up from the end to the
             * beginning of the episode (of course still dependent on the
             * discount of the model).
             *
             * The lambda parameter must be >= 0.0 and <= 1.0, otherwise the
             * function will throw an std::invalid_argument.
             *
             * @param l The new lambda parameter.
             */
            void setLambda(double l);

            /**
             * @brief This function returns the currently set lambda parameter.
             *
             * @return The currently set lambda parameter.
             */
            double getLambda() const;

            /**
             * @brief This function sets the trace cutoff parameter.
             *
             * This parameter determines when a trace is removed, as its
             * coefficient has become too small to bother updating its value.
             *
             * Note that the trace cutoff is performed on the overall
             * discount*lambda value, and not only on lambda. So this parameter
             * is useful even when lambda is 1.
             *
             * @param t The new trace cutoff value.
             */
            void setTolerance(double t);

            /**
             * @brief This function returns the currently set trace cutoff parameter.
             *
             * @return The currently set trace cutoff parameter.
             */
            double getTolerance() const;

            /**
             * @brief This function clears the already set traces.
             */
            void clearTraces();

            /**
             * @brief This function returns the currently set traces.
             *
             * @return The currently set traces.
             */
            const Traces & getTraces() const;

            /**
             * @brief This function sets the currently set traces.
             *
             * This method is provided in case you have a need to tinker with
             * the internal traces. You generally don't unless you are building
             * on top of SARSAL in order to do something more complicated.
             *
             * @param t The currently set traces.
             */
            void setTraces(const Traces & t);

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
             * been computed elsewhere. SARSAL will then continue building upon
             * it.
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
            double lambda_, tolerance_;
            // This is used to avoid multiplying the discount and lambda all the time.
            double gammaL_;

            QFunction q_;
            Traces traces_;
    };

    template <typename M, typename>
    SARSAL::SARSAL(const M& model, const double alpha, const double lambda, const double tolerance) :
            SARSAL(model.getS(), model.getA(), model.getDiscount(), alpha, lambda, tolerance) {}
}
#endif
