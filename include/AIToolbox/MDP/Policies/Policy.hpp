#ifndef AI_TOOLBOX_MDP_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_POLICY_HEADER_FILE

#include <vector>
#include <tuple>

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/MDP/Policies/PolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class represents an MDP Policy.
         *
         * This class is one of the many ways to represent an MDP Policy. In
         * particular, it maintains a 2 dimensional table of probabilities
         * determining the probability of choosing an action in a given state.
         *
         * The class offers facilities to sample from these distributions, so
         * that you can directly embed it into a decision-making process.
         *
         * Building this object is somewhat expensive, so it should be done
         * mostly when it is known that the final solution won't change again.
         * Otherwise you may want to build a wrapper around some data to
         * extract the policy dynamically.
         */
        class Policy : public PolicyInterface {
            public:
                using PolicyTable = Table2D;

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor initializes the internal policy table so that
                 * each action in each state has the same probability of being
                 * chosen (random policy). This class guarantees that at any point
                 * the internal policy is a true probability distribution, i.e.
                 * for each state the sum of the probabilities of choosing an action
                 * sums up to 1.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 */
                Policy(size_t s, size_t a);

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor simply copies policy probability values
                 * from any other compatible PolicyInterface, and stores them
                 * internally. This is probably the main way you may want to use
                 * this class.
                 *
                 * This may be a useful thing to do in case the policy that is
                 * being copied is very costly to use (for example, QGreedyPolicy)
                 * and it is known that it will not change anymore.
                 *
                 * @param p The policy which is being copied.
                 */
                Policy(const PolicyInterface & p);

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor copies the implied policy contained in a ValueFunction.
                 * Keep in mind that the policy stored within a ValueFunction is
                 * non-stochastic in nature, since for each state it can only
                 * save a single action.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param v The ValueFunction used as a basis for the Policy.
                 */
                Policy(size_t s, size_t a, const ValueFunction & v);

                /**
                 * @brief This function chooses a random action for state s, following the policy distribution.
                 *
                 * @param s The sampled state of the policy.
                 *
                 * @return The chosen action.
                 */
                virtual size_t sampleAction(const size_t & s) const override;

                /**
                 * @brief This function returns the probability of taking the specified action in the specified state.
                 *
                 * @param s The selected state.
                 * @param a The selected action.
                 *
                 * @return The probability of taking the selected action in the specified state.
                 */
                virtual double getActionProbability(const size_t & s, const size_t & a) const override;

                /**
                 * @brief This function sets the policy for a particular state.
                 *
                 * This function copies correctly sized container into the policy,
                 * normalizing it so that it sums to 1. If the size of the container
                 * is incorrect, an std::invalid_argument is thrown.
                 *
                 * The container needs to support size() and begin/end iterators.
                 * The elements of the container must be convertible to double, and the
                 * results of these conversions MUST BE positive.
                 *
                 * @tparam T The type of the input container.
                 * @param s The state where the policy is being set.
                 * @param container The input container.
                 */
                template <typename T>
                void setStatePolicy(size_t s, const T & container);

                /**
                 * @brief This function returns a copy of a particular slice of the policy.
                 *
                 * @param s The requested state.
                 *
                 * @return The probabilities of choosing each action in state s.
                 */
                std::vector<double> getStatePolicy(size_t s) const;

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
                void setStatePolicy(size_t s, size_t a);

                /**
                 * @brief This function enables inspection of the internal policy.
                 *
                 * @return A constant reference to the internal policy.
                 */
                const PolicyTable & getPolicyTable() const;

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
                PolicyTable policy_;

                friend std::istream& operator>>(std::istream &is, Policy & p);
        };

        /**
         * @brief This function reads a policy from a file.
         *
         * This function reads files that have been outputted through
         * operator<<(). If not enough values can be extracted from
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
        void Policy::setStatePolicy(const size_t s, const T & container) {
            if ( container.size() != getA() ) throw std::invalid_argument("Container to copy has the wrong size.");

            auto ref = policy_[s]; // This is needed because policy_[s] by itself is a temporary. We don't want begin and end to be of different things.
            normalizeProbability(std::begin(container), std::end(container), std::begin(ref));
        }
    }
}

#endif
