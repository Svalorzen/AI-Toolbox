#ifndef AI_TOOLBOX_MDP_EPSILON_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_EPSILON_POLICY_HEADER_FILE

#include <AIToolbox/EpsilonPolicyInterface.hpp>

namespace AIToolbox {
    namespace MDP {
        class EpsilonPolicy : public EpsilonPolicyInterface<size_t, size_t, size_t> {
            public:
                using Base = EpsilonPolicyInterface<size_t, size_t, size_t>;

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
                EpsilonPolicy(const Base::Base & p, double epsilon = 0.9);

            protected:
                /**
                 * @brief This function returns a random action in the Action space.
                 *
                 * @return A valid random action.
                 */
                virtual size_t sampleRandomAction() const;

                // Used to sampled random actions
                mutable std::uniform_int_distribution<size_t> randomDistribution_;
        };
    }
}

#endif
