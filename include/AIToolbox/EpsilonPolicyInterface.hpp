#ifndef AI_TOOLBOX_EPSILON_POLICY_INTERFACE_HEADER_FILE
#define AI_TOOLBOX_EPSILON_POLICY_INTERFACE_HEADER_FILE

#include <AIToolbox/PolicyInterface.hpp>

#include <stdexcept>

namespace AIToolbox {
    /**
     * @brief This class is a policy wrapper for epsilon action choice.
     *
     * This class is used to wrap already existing policies to implement
     * automatic exploratory behaviour (e.g. epsilon-greedy policies).
     *
     * An epsilon-greedy policy is a policy that takes a greedy action a
     * certain percentage of the time, and otherwise takes a random action.
     * They are useful to force the agent to explore an unknown model, in order
     * to gain new information to refine it and thus gain more reward.
     *
     * Please note that to obtain an epsilon-greedy policy the wrapped
     * policy needs to already be greedy with respect to the model.
     *
     * @tparam State This defines the type that is used to store the state space.
     * @tparam Sampling This defines the type that is used to sample from the policy.
     * @tparam Action This defines the type that is used to handle actions.
     */
    template <typename State, typename Sampling, typename Action>
    class EpsilonPolicyInterface : public PolicyInterface<State, Sampling, Action> {
        public:
            using Base = PolicyInterface<State, Sampling, Action>;
            /**
             * @brief Basic constructor.
             *
             * This constructor saves the input policy and the epsilon
             * parameter for later use.
             *
             * The epsilon parameter must be >= 0.0 and <= 1.0,
             * otherwise the constructor will throw an std::invalid_argument.
             *
             * @param p The policy that is being extended.
             * @param epsilon The parameter that controls the amount of exploration.
             */
            EpsilonPolicyInterface(const Base & p, double epsilon = 0.9);

            /**
             * @brief This function chooses a random action for state s, following the policy distribution and epsilon.
             *
             * This function has a probability of (1 - epsilon) of selecting
             * a random action. Otherwise, it selects an action according
             * to the distribution specified by the wrapped policy.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction(const Sampling & s) const override;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * This function takes into account parameter epsilon
             * while computing the final probability.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const Sampling & s, const Action & a) const override;

            /**
             * @brief This function sets the epsilon parameter.
             *
             * The epsilon parameter determines the amount of exploration this
             * policy will enforce when selecting actions. In particular
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
            /**
             * @brief This function returns a random action in the Action space.
             *
             * @return A valid random action.
             */
            virtual Action sampleRandomAction() const = 0;

            /**
             * @brief This function returns the probability of picking a random action.
             *
             * This is simply one over the action space, but since the action
             * space may not be a single number we leave to implementation to
             * decide how to best compute this.
             *
             * @return The probability of picking an an action at random.
             */
            virtual double getRandomActionProbability() const = 0;

            const Base & policy_;
            double epsilon_;

            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };

    template <typename State, typename Sampling, typename Action>
    EpsilonPolicyInterface<State, Sampling, Action>::EpsilonPolicyInterface(const Base & p, const double e) :
            Base(p.getS(), p.getA()), policy_(p), epsilon_(e),
            sampleDistribution_(0.0, 1.0)
    {
        if ( epsilon_ < 0.0 || epsilon_ > 1.0 ) throw std::invalid_argument("Epsilon must be >= 0 and <= 1");
    }

    template <typename State, typename Sampling, typename Action>
    Action EpsilonPolicyInterface<State, Sampling, Action>::sampleAction(const Sampling & s) const {
        const double pe = sampleDistribution_(this->rand_);
        if ( pe > epsilon_ )
            return sampleRandomAction();

        return policy_.sampleAction(s);
    }

    template <typename State, typename Sampling, typename Action>
    double EpsilonPolicyInterface<State, Sampling, Action>::getActionProbability(const Sampling & s, const Action & a) const {
        //          Probability of taking old decision          Other probability
        return epsilon_ * policy_.getActionProbability(s,a) + ( 1.0 - epsilon_ ) * getRandomActionProbability();
    }

    template <typename State, typename Sampling, typename Action>
    void EpsilonPolicyInterface<State, Sampling, Action>::setEpsilon(const double e) {
        if ( e < 0.0 || e > 1.0 ) throw std::invalid_argument("Epsilon parameter was outside specified bounds");
        epsilon_ = e;
    }

    template <typename State, typename Sampling, typename Action>
    double EpsilonPolicyInterface<State, Sampling, Action>::getEpsilon() const {
        return epsilon_;
    }
}

#endif
