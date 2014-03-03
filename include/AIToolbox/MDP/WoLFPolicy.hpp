#ifndef AI_TOOLBOX_MDP_WOLF_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_WOLF_POLICY_HEADER_FILE

#include <vector>

#include <AIToolbox/MDP/QPolicyInterface.hpp>
#include <AIToolbox/Policy.hpp>

namespace AIToolbox {
    namespace MDP {

        class WoLFPolicy : public QPolicyInterface {
            public:
                WoLFPolicy(const QFunction & q, double deltaw = 0.0125, double deltal = 0.05);

                /**
                 * @brief This function updates the WoLF policy based on changes in the QFunction.
                 *
                 * @param s The state that needs to be updated.
                 */
                void updatePolicy(size_t s);

                /**
                 * @brief This function chooses an action for state s, following the policy distribution.
                 *
                 * @param s The sampled state of the policy.
                 *
                 * @return The chosen action.
                 */
                virtual size_t sampleAction(size_t s) const override;

                /**
                 * @brief This function returns the probability of taking the specified action in the specified state.
                 *
                 * @param s The selected state.
                 * @param a The selected action.
                 *
                 * @return The probability of taking the selected action in the specified state.
                 */
                virtual double getActionProbability(size_t s, size_t a) const override;

            private:
                double deltaW_, deltaL_;

                std::vector<unsigned> c_;
                Policy avgPolicy_, actualPolicy_;
        };

    }
}

#endif
