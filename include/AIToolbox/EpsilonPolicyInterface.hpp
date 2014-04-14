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
     * Please note that to obtain an epsilon-greedy policy the wrapped
     * policy needs to already be greedy with respect to the model.
     */
    template <typename State>
    class EpsilonPolicyInterface : public PolicyInterface<State> {
        public:
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
            EpsilonPolicyInterface(const PolicyInterface<State> & p, double epsilon = 0.9);

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
            virtual size_t sampleAction(const State & s) const override;

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
            virtual double getActionProbability(const State & s, size_t a) const override;

            /**
             * @brief This function sets the epsilon parameter.
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
            const PolicyInterface<State> & policy_;
            double epsilon_;

            // Used to sampled random actions
            mutable std::uniform_int_distribution<size_t> randomDistribution_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };

    template <typename State>
    EpsilonPolicyInterface<State>::EpsilonPolicyInterface(const PolicyInterface<State> & p, double e) : PolicyInterface<State>(p.getS(), p.getA()), policy_(p), epsilon_(e), randomDistribution_(0, this->A-1), sampleDistribution_(0.0,1.0)
    {
        if ( epsilon_ < 0.0 || epsilon_ > 1.0 ) throw std::invalid_argument("Epsilon must be >= 0 and <= 1");
    }

    template <typename State>
    size_t EpsilonPolicyInterface<State>::sampleAction(const State & s) const {
        double pe = sampleDistribution_(this->rand_);
        if ( pe > epsilon_ ) {
            return randomDistribution_(this->rand_);
        }
        return policy_.sampleAction(s);
    }

    template <typename State>
    double EpsilonPolicyInterface<State>::getActionProbability(const State & s, size_t a) const {
        //          Probability of taking old decision          Other probability
        return epsilon_ * policy_.getActionProbability(s,a) + ( 1.0 - epsilon_ ) / this->A;
    }

    template <typename State>
    void EpsilonPolicyInterface<State>::setEpsilon(double e) {
        if ( e < 0.0 || e > 1.0 ) throw std::invalid_argument("Epsilon parameter was outside specified bounds");
        epsilon_ = e;
    }

    template <typename State>
    double EpsilonPolicyInterface<State>::getEpsilon() const {
        return epsilon_;
    }
}

#endif
