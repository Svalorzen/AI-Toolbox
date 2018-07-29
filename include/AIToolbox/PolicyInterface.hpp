#ifndef AI_TOOLBOX_POLICYINTERFACE_HEADER_FILE
#define AI_TOOLBOX_POLICYINTERFACE_HEADER_FILE

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox {
    /**
     * @brief This class represents the base interface for policies.
     *
     * This class represents an interface that all policies must conform to.
     * The interface is generic as different methods may have very different
     * ways to store and compute policies, and this interface simply asks
     * for a way to sample them.
     *
     * This class is templatized in order to work with as many problem classes
     * as possible. It doesn't really implement any specific functionality, but
     * should be used as a guide in order to decide which methods a new policy
     * should have.
     *
     * The State and Sampling parameters are separate since in POMDPs we still
     * have "integer" states, but we often want to sample from beliefs. Having
     * two separate parameters allows for this. In the MDP case they are both
     * the same.
     *
     * @tparam State This defines the type that is used to store the state space.
     * @tparam Sampling This defines the type that is used to sample from the policy.
     * @tparam Action This defines the type that is used to handle actions.
     */
    template <typename State, typename Sampling, typename Action>
    class PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            PolicyInterface(State s, Action a);

            /**
             * @brief Basic virtual destructor.
             */
            virtual ~PolicyInterface();

            /**
             * @brief This function chooses a random action for state s, following the policy distribution.
             *
             * @param s The sampled state of the policy.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction(const Sampling & s) const = 0;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const Sampling & s, const Action & a) const = 0;

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
            const State & getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            const Action & getA() const;

        protected:
            State S;
            Action A;

            // This is mutable because sampling doesn't really change the policy
            mutable RandomEngine rand_;
    };

    template <typename State, typename Sampling, typename Action>
    PolicyInterface<State, Sampling, Action>::PolicyInterface(State s, Action a) :
            S(std::move(s)), A(std::move(a)), rand_(Impl::Seeder::getSeed()) {}

    template <typename State, typename Sampling, typename Action>
    PolicyInterface<State, Sampling, Action>::~PolicyInterface() {}

    template <typename State, typename Sampling, typename Action>
    const State & PolicyInterface<State, Sampling, Action>::getS() const { return S; }

    template <typename State, typename Sampling, typename Action>
    const Action & PolicyInterface<State, Sampling, Action>::getA() const { return A; }

    /**
     * @brief This class represents the base interface for policies in games and bandits.
     *
     * This specialization simply removes the states from the PolicyInterface,
     * since in normal games and bandits there is no state, and we can simplify
     * the interface. The rest is the same.
     *
     * @tparam Action This defines the type that is used to handle actions.
     */
    template <typename Action>
    class PolicyInterface<void, void, Action> {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param a The number of actions available to the agent.
             */
            PolicyInterface(Action a);

            /**
             * @brief Basic virtual destructor.
             */
            virtual ~PolicyInterface();

            /**
             * @brief This function chooses a random action, following the policy distribution.
             *
             * @return The chosen action.
             */
            virtual Action sampleAction() const = 0;

            /**
             * @brief This function returns the probability of taking the specified action.
             *
             * @param a The selected action.
             *
             * @return The probability of taking the selected action.
             */
            virtual double getActionProbability(const Action & a) const = 0;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            const Action & getA() const;

        protected:
            Action A;

            // This is mutable because sampling doesn't really change the policy
            mutable RandomEngine rand_;
    };

    template <typename Action>
    PolicyInterface<void, void, Action>::PolicyInterface(Action a) :
            A(std::move(a)), rand_(Impl::Seeder::getSeed()) {}

    template <typename Action>
    PolicyInterface<void, void, Action>::~PolicyInterface() {}

    template <typename Action>
    const Action & PolicyInterface<void, void, Action>::getA() const { return A; }
}

#endif
