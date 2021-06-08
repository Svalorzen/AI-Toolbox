#ifndef AI_TOOLBOX_MDP_ROLLOUT_HEADER_FILE
#define AI_TOOLBOX_MDP_ROLLOUT_HEADER_FILE

#include <AIToolbox/MDP/Types.hpp>
#include <AIToolbox/MDP/TypeTraits.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <unordered_map>

namespace AIToolbox::MDP {
    /**
     * @brief This function performs a rollout from the input state.
     *
     * This function performs a rollout until the agent either reaches the
     * desired depth, or reaches a terminal state. The overall return is
     * finally returned, from the point of the input state, and with the future
     * rewards discounted appropriately.
     *
     * This function is generally used in Monte-Carlo tree search-like
     * algorithms, like MCTS or POMCP, to speed up discovery of promising
     * actions without necessarily expanding their search tree. This avoids
     * wasting lots of computation and memory on states far from our root that
     * we will probably never see again, while at the same time still getting
     * an estimate for the rest of the simulation.
     *
     * @param m The model to use for the rollout.
     * @param s The state to start the rollout from.
     * @param maxDepth The maximum number of timesteps to look into the future.
     * @param rnd A random number generator.
     *
     * @return The discounted return from the input state.
     */
    template <typename M, typename Gen>
    double rollout(const M & m, size_t s, const unsigned maxDepth, Gen & rnd) {
        static_assert(is_generative_model_v<M> || is_generative_model_variable_actions_v<M>, "rollout() only works for generative MDP models!");

        double rew = 0.0, totalRew = 0.0, gamma = 1.0;

        // Here we have two separate branches depending on whether the model
        // provides a variable number of actions or not. Note that they are
        // nearly identical, and the only difference is how we instantiate the
        // distribution from which to randomly sample actions.

        if constexpr (is_generative_model_v<M>) {
            // If we don't have variable actions, we instantiate the
            // distribution outside of the loop for a slight performance
            // increase.
            std::uniform_int_distribution<size_t> dist(0, m.getA()-1);
            for (unsigned depth = 0; depth < maxDepth; ++depth ) {
                std::tie( s, rew ) = m.sampleSR( s, dist(rnd) );
                totalRew += gamma * rew;

                if (m.isTerminal(s))
                    return totalRew;

                gamma *= m.getDiscount();
            }
        } else {
            // Otherwise, we need to poll the model at every timestep to check
            // the allowed number of actions, and sample from those.
            for (unsigned depth = 0; depth < maxDepth; ++depth ) {
                std::uniform_int_distribution<size_t> dist(0, m.getA(s)-1);
                std::tie( s, rew ) = m.sampleSR( s, dist(rnd) );
                totalRew += gamma * rew;

                if (m.isTerminal(s))
                    return totalRew;

                gamma *= m.getDiscount();
            }
        }
        return totalRew;
    }
}

#endif
