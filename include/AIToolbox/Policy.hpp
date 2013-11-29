#ifndef AI_TOOLBOX_POLICY_HEADER_FILE
#define AI_TOOLBOX_POLICY_HEADER_FILE

#include <cstddef>
#include <vector>
#include <tuple>
#include <random>

#include <boost/multi_array.hpp>
#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    /**
     * @brief This class represents a policy.
     */
    class Policy {
        public:
            using PolicyTable = Table2D;

            /**
             * @brief Basic constrctor.
             *
             * This constructor initializes the internal policy table so that
             * each action in each state has the same probability of being 
             * chosen (random policy). This class guarantees that at any point
             * the internal policy is a true probability distribution, i.e.
             * for each state the sum of the probabilities of chosing an action
             * sums up to 1.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            Policy(size_t s, size_t a);

            /**
             * @brief This function returns a copy of a particular slice of the policy.
             *
             * @param s The requested state.
             *
             * @return The probabilities of choosing each action in state s.
             */
            std::vector<double> getStatePolicy(size_t s) const;

            /**
             * @brief This function chooses a random action for state s, following the policy distribution.
             *
             * The epsilon parameter controls the randomness of the choice.
             * If epsilon = 1 or greater, the action returned follows the distribution
             * specified by the policy. If epsilon = 0 or lower, the action returned is
             * random. A value of 0.9 would mean a 0.9 chance of getting a policy action,
             * and a 0.1 chance of getting a random action.
             *
             * @param s The sampled state of the policy.
             * @param epsilon The epsilon-greediness of the policy.
             *
             * @return The chosen action.
             */
            size_t getAction(size_t s, double epsilon = 1.0) const;

            /**
             * @brief This function sets the policy for a particular state.
             *
             * This function copies correctly sized container into the policy,
             * normalizing it so that it sums to 1. If the size of the container 
             * is incorrect, nothing happens.
             *
             * The container needs to support size() and begin/end iterators.
             *
             * @tparam T The type of the input container.
             * @param s The state where the policy is being set.
             * @param container The input container.
             */
            template <typename T>
            void setPolicy(size_t s, const T & container);

            /**
             * @brief This function sets the policy for a particular state.
             *
             * This function represents an easier way to set a probability
             * of 1 to execute a given action in a particular state, and 
             * setting all other probabilities to 0.
             *
             * @param s The state where the policy is being set.
             * @param a The action that must be chosen in state s.
             */
            void setPolicy(size_t s, size_t a);

            /**
             * @brief This function enables inspection of the internal policy.
             *
             * @return A constant reference to the internal policy.
             */
            const PolicyTable & getPolicy() const;

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

            /**
             * @brief Prints the policy to a stream.
             *
             * This function differs from operator<<() in that it
             * avoids printing all probabilities that equal 0. This
             * results in a much more readable and smaller file.
             * 
             * The format of the output file is:
             *
             * state_number action_number probability
             *
             * @param os The stream where the policy is printed.
             */
            void prettyPrint(std::ostream & os) const;
        private:
            size_t S, A;
            PolicyTable policy_;

            // These are mutable because sampling doesn't really change the MDP
            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
            mutable std::uniform_int_distribution<int> randomDistribution_;

            friend std::istream& operator>>(std::istream &is, Policy &);
    };

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
    std::ostream& operator<<(std::ostream &os, const Policy & p);

    /**
     * @brief This function reads a policy from a file.
     *
     * This function reads files that have been outputted through
     * operator>>(). If not enough values can be extracted from
     * the stream, the function stops and the input policy is
     * not modified. In addition, it checks whether the probability
     * values are within 0 and 1. 
     * 
     * State and actions are also verified, and this function does
     * not accept a randomly shuffled policy file. The file must
     * be sorted by state, and each state must be sorted by action.
     *
     * As a layer of additional precaution, the function normalizes
     * the policy once it has been read, to assure true probability
     * distribution on the internal policy.
     *
     * @param is The stream were the policy is being read from.
     * @param p The policy that is being assigned.
     *
     * @return The input stream.
     */
    std::istream& operator>>(std::istream &is, Policy & p);

    template <typename T>
    void Policy::setPolicy(size_t s, const T & container) {
        if ( container.size() != A ) return;

        double norm = static_cast<double>(std::accumulate(std::begin(container), std::end(container), 0.0));
        std::transform(std::begin(container), std::end(container), std::begin(policy_[s]), [norm](decltype(*begin(container)) t){ return t/norm; });
    }

}

#endif
