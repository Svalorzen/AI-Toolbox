#ifndef AI_TOOLBOX_MDP_OFF_POLICY_TEMPLATE_HEADER_FILE
#define AI_TOOLBOX_MDP_OFF_POLICY_TEMPLATE_HEADER_FILE

#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This class contains all the boilerplates for off-policy methods.
     */
    class OffPolicyBase {
        public:
            using Trace = std::tuple<size_t, size_t, double>;
            using Traces = std::vector<Trace>;

            /**
             * @brief Basic construtor.
             *
             * @param s The size of the state space.
             * @param a The size of the action space.
             * @param discount The discount of the environment.
             * @param alpha The learning rate.
             * @param tolerance The cutoff point for eligibility traces.
             */
            OffPolicyBase(size_t s, size_t a, double discount = 1.0, double alpha = 0.1, double tolerance = 0.001);

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
             * accordingly. The final behaviour is very dependent on this
             * parameter.
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
             * The discount parameter controls how much we care about future
             * rewards. If 1, then any reward is the same, if obtained now or
             * in a million timesteps. Thus the algorithm will optimize overall
             * reward accretion. When less than 1, rewards obtained in the
             * presents are valued more than future rewards.
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
             * @brief This function sets the trace cutoff parameter.
             *
             * This parameter determines when a trace is removed, as its
             * coefficient has become too small to bother updating its value.
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
             * on top of this class in order to do something more complicated.
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
             * been computed elsewhere.
             *
             * @param qfun The new QFunction to set.
             */
            void setQFunction(const QFunction & qfun);

        protected:
            size_t S, A;
            double discount_, alpha_, tolerance_;

            /**
             * @brief This function updates the traces using the input data.
             *
             * This operation is basically identical to what SARSAL does.
             *
             * \sa SARSAL::stepUpdateQ
             *
             * @param s The state we were before.
             * @param a The action we did.
             * @param error The error used to update the QFunction.
             * @param traceDiscount The discount for all traces in memory.
             */
            void updateTraces(size_t s, size_t a, double error, double traceDiscount);

            QFunction q_;
            Traces traces_;
    };

    /**
     * @brief This class is a general version of off-policy evaluation.
     *
     * This class is used to compute the QFunction of a given policy, when you
     * are actually acting and gathering data following *another* policy (which
     * is why it's called off-policy).
     *
     * Keep in mind that these kind of methods are not very efficient when
     * either the target or the behaviour policy are very deterministic. This
     * is because greedy policies (at least with methods that use some kind of
     * importance sampling) tend to cut traces short, which is basically
     * equivalent to discarding data (this must be done to ensure correctness
     * though).
     *
     * Note that this class does not necessarily encompass all off-policy
     * evaluation methods. It only does for the one that use eligibility traces
     * in a certain form, such as ImportanceSampling, RetraceLambda, etc.
     *
     * This class is supposed to be used as a CRTP parent. The child must
     * derive it as:
     *
     * ```
     *     class Child : public OffPolicyEvaluation<Child> {};
     * ```
     *
     * In addition, the child must define the function
     *
     * ```
     *     double getTraceDiscount(size_t s, size_t a, size_t s1, double rew) const;
     * ```
     *
     * This will then be automatically called here to compute the amount to
     * decrease the traces during the stepUpdateQ. For example, in
     * ImportanceSampling the function would return:
     *
     * ```
     *     return target_.getActionProbability(s, a) / behaviour_.getActionProbability(s, a);
     * ```
     */
    template <typename Derived>
    class OffPolicyEvaluation : public OffPolicyBase {
        public:
            using Parent = OffPolicyBase;

            /**
             * @brief Basic constructor.
             *
             * @param target The policy to be evaluated.
             * @param discount The discount of the environment.
             * @param alpha The learning rate parameter.
             * @param tolerance The trace cutoff parameter.
             */
            OffPolicyEvaluation(const PolicyInterface & target, double discount = 1.0,
                    double alpha = 0.1, double tolerance = 0.001);

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
            void stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew);

        protected:
            const PolicyInterface & target_;
    };

    /**
     * @brief This class is a general version of off-policy control.
     *
     * This class is used to compute the optimal QFunction, when you are
     * actually acting and gathering data following *another* policy (which is
     * why it's called off-policy). This is what QLearning does, for example.
     *
     * As in the off-policy evaluation case, this method does not work well
     * with a deterministic behaviour. Even worse, we're trying to find out the
     * optimal policy, which is greedy by definition. Thus, this method assumes
     * that the target is an epsilon greedy policy, and needs to know its
     * epsilon. You should, over time, decrease the epsilon and this method
     * should converge to the optimal QFunction.
     *
     * Note that this class does not necessarily encompass all off-policy
     * control methods. It only does for the one that use eligibility traces
     * in a certain form, such as ImportanceSampling, RetraceLambda, etc.
     *
     * This class is supposed to be used as a CRTP parent. The child must
     * derive it as:
     *
     * ```
     *     class Child : public OffPolicyEvaluation<Child> {};
     * ```
     *
     * In addition, the child must define the function
     *
     * ```
     *     double getTraceDiscount(size_t s, size_t a, size_t s1, double rew, size_t maxA) const;
     * ```
     *
     * Where maxA is the already computed best greedy action for state s.
     *
     * This will then be automatically called here to compute the amount to
     * decrease the traces during the stepUpdateQ. For example, in
     * ImportanceSampling the function would return:
     *
     * ```
     *     const auto prob = epsilon_ / A + (maxA == a) * (1.0 - epsilon_);
     *     return prob / behaviour_.getActionProbability(s, a);
     * ```
     *
     * Note how this is different from the OffPolicyEvaluation case, as we
     * assume the target policy to be epsilon greedy.
     */
    template <typename Derived>
    class OffPolicyControl : public OffPolicyBase {
        public:
            using Parent = OffPolicyBase;

            /**
             * @brief Basic constructor.
             *
             * @param s The size of the state space.
             * @param a The size of the action space.
             * @param discount The discount of the environment.
             * @param alpha The learning rate parameter.
             * @param tolerance The trace cutoff parameter.
             * @param epsilon The epsilon of the implied target greedy epsilon policy.
             */
            OffPolicyControl(size_t s, size_t a, double discount = 1.0, double alpha = 0.1,
                        double tolerance = 0.001, double epsilon = 0.1);

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
            void stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew);

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter determines the amount of epsilon
             * this policy will enforce when selecting actions. In particular
             * actions are going to selected randomly with probability
             * (1-epsilon), and are going to be selected following the
             * underlying policy with probability epsilon.
             *
             * The epsilon parameter must be >= 0.0 and <= 1.0,
             * otherwise the function will do throw std::invalid_argument.
             *
             * @param e The new epsilon parameter.
             */
            void setEpsilon(double e);

            /**
             * @brief This function will return the currently set epsilon parameter.
             *
             * @return The currently set epsilon parameter.
             */
            double getEpsilon() const;

        protected:
            double epsilon_;
    };

    template <typename Derived>
    void OffPolicyEvaluation<Derived>::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        auto expectedQ = 0.0;
        for (size_t a = 0; a < A; ++a)
            expectedQ += q_(s1, a) * target_.getActionProbability(s1, a);

        const auto error = alpha_ * ( rew + discount_ * expectedQ - q_(s, a) );
        const auto traceDiscount = discount_ * static_cast<Derived*>(this)->getTraceDiscount(s, a, s1, rew);

        updateTraces(s, a, error, traceDiscount);
    }

    template <typename Derived>
    void OffPolicyControl<Derived>::stepUpdateQ(const size_t s, const size_t a, const size_t s1, const double rew) {
        // The basic idea here is that differently from the evaluation, we want
        // to do a maximization. At the same time, to work we need to "round
        // the edges" a bit, and that's why we assume an epsilon-greedy policy.
        //
        // The expected value we can compute easily since each action has the
        // same probability of being chosen, but for the greedy one which is
        // more likely.
        size_t maxA;
        double expectedQ = 0.0;
        double maxV = std::numeric_limits<double>::lowest();
        for (size_t aa = 0; aa < A; ++aa) {
            expectedQ += q_(s1, aa);
            if (maxV < q_(s1, aa)) {
                maxA = aa;
                maxV = q_(s1, aa);
            }
        }
        expectedQ *= epsilon_ / A;
        expectedQ += (1.0 - epsilon_) * maxV;

        const auto error = alpha_ * ( rew + discount_ * expectedQ - q_(s, a) );
        const auto traceDiscount = discount_ * static_cast<Derived*>(this)->getTraceDiscount(s, a, s1, rew, maxA);

        updateTraces(s, a, error, traceDiscount);
    }

    template <typename Derived>
    OffPolicyEvaluation<Derived>::OffPolicyEvaluation(
        const PolicyInterface & target,
        const double discount, const double alpha, const double tolerance
    ) :
        Parent(target.getS(), target.getA(), discount, alpha, tolerance),
        target_(target) {}

    template <typename Derived>
    OffPolicyControl<Derived>::OffPolicyControl(
        const size_t s, const size_t a, const double discount,
        const double alpha, const double tolerance, const double epsilon
    ) :
        Parent(s, a, discount, alpha, tolerance)
    {
        setEpsilon(epsilon);
    }

    template <typename Derived>
    void OffPolicyControl<Derived>::setEpsilon(const double e) {
        if ( e < 0.0 || e > 1.0 ) throw std::invalid_argument("Epsilon must be >= 0 and <= 1");
        epsilon_ = e;
    }

    template <typename Derived>
    double OffPolicyControl<Derived>::getEpsilon() const {
        return epsilon_;
    }
}

#endif
