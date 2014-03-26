#ifndef AI_TOOLBOX_POLICYINTERFACE_HEADER_FILE
#define AI_TOOLBOX_POLICYINTERFACE_HEADER_FILE

#include <cstddef>
#include <random>

#include <iosfwd>
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
     * This class is templatized since it works as an interface for both
     * MDP and POMDP policies. In the case of MDPs, the template parameter
     * State is of type size_t, which represents the states from which we are
     * sampling. In case of POMDPs, the template parameter is of type Belief,
     * which allows us to sample the policy from different beliefs.
     *
     * @tparam State This defines the type that is used to sample from the policy.
     */
    template <typename State>
    class PolicyInterface {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            PolicyInterface(size_t s, size_t a);

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
            virtual size_t sampleAction(const State & s) const = 0;

            /**
             * @brief This function returns the probability of taking the specified action in the specified state.
             *
             * @param s The selected state.
             * @param a The selected action.
             *
             * @return The probability of taking the selected action in the specified state.
             */
            virtual double getActionProbability(const State & s, size_t a) const = 0;

            /**
             * @brief This function returns the number of states of the world.
             *
             * @return The total number of states.
             */
            size_t getS() const;

            /**
             * @brief This function returns the number of available actions to the agent.
             *
             * @return The total number of actions.
             */
            size_t getA() const;
        protected:
            size_t S, A;

            // These are mutable because sampling doesn't really change the policy
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };

    template <typename State>
    PolicyInterface<State>::PolicyInterface(size_t s, size_t a) : S(s), A(a),
                                                           rand_(Impl::Seeder::getSeed()),
                                                           sampleDistribution_(0.0, 1.0) {}

    template <typename State>
    PolicyInterface<State>::~PolicyInterface() {}

    template <typename State>
    size_t PolicyInterface<State>::getS() const { return S; }
    template <typename State>
    size_t PolicyInterface<State>::getA() const { return A; }

    /**
     * @brief This function prints the whole policy to a file.
     *
     * This function outputs each and every value of the policy
     * for easy parsing. The output is broken into multiple lines
     * where each line is of the format:
     *
     * state_number action_number probability
     *
     * And all lines are sorted by state, and each state is sorted
     * by action.
     *
     * @param os The stream where the policy is printed.
     * @param p The policy that is begin printed.
     *
     * @return The original stream.
     */
    std::ostream& operator<<(std::ostream &os, const PolicyInterface<size_t> & p);
}

#endif
