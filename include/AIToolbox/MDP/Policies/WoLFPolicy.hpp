#ifndef AI_TOOLBOX_MDP_WOLF_POLICY_HEADER_FILE
#define AI_TOOLBOX_MDP_WOLF_POLICY_HEADER_FILE

#include <vector>

#include <AIToolbox/MDP/Policies/QPolicyInterface.hpp>
#include <AIToolbox/MDP/Policies/Policy.hpp>

namespace AIToolbox {
    namespace MDP {

        /**
         * @brief This class models the WoLF learning algorithm.
         *
         * What this algorithm does is it progressively modifies the policy
         * given changes in the underlying QFunction. In particular, it
         * modifies it rapidly if the agent is "losing" (getting less reward
         * than expected), and more slowly when "winning", since there's little
         * reason to change behaviour when things go right.
         *
         * An advantage of this algorithm is that it can allow the policy to
         * converge to non-deterministic solutions: for example two players
         * trying to outmatch each other in rock-paper-scissor. At the same
         * time, this particular version of the algorithm can take quite some
         * time to converge to a good solution.
         */
        class WoLFPolicy : public QPolicyInterface {
            public:
                WoLFPolicy(const QFunction & q, double deltaw = 0.0125, double deltal = 0.05);

                /**
                 * @brief This function updates the WoLF policy based on changes in the QFunction.
                 *
                 * This function should be called between agent's actions,
                 * using the agent's current state.
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
                virtual size_t sampleAction(const size_t & s) const override;

                /**
                 * @brief This function returns the probability of taking the specified action in the specified state.
                 *
                 * @param s The selected state.
                 * @param a The selected action.
                 *
                 * @return The probability of taking the selected action in the specified state.
                 */
                virtual double getActionProbability(const size_t & s, size_t a) const override;

            private:
                double deltaW_, deltaL_;

                std::vector<unsigned> c_;
                Policy avgPolicy_, actualPolicy_;
        };

    }
}

#endif
