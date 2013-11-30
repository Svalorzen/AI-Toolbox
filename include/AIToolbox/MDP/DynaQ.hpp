#ifndef AI_TOOLBOX_MDP_DYNAQ_HEADER_FILE
#define AI_TOOLBOX_MDP_DYNAQ_HEADER_FILE

#include <AIToolbox/MDP/DynaQInterface.hpp>

#include <boost/functional/hash.hpp>
#include <utility>
#include <unordered_set>
#include <vector>

namespace AIToolbox {
    namespace MDP {
        class DynaQ : public DynaQInterface {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param discount The discount of the QLearning method.
                 * @param n The number of sampling passes to do on the model upon batchUpdateQ().
                 */
                DynaQ(size_t s, size_t a, double discount = 0.9, unsigned n = 50);

                /**
                 * @brief This function is updates DynaQ sampling list.
                 *
                 * The sampling list in DynaQ is a simple list of all visited
                 * state action pairs. This function is responsible for inserting
                 * them in a set, keeping them unique. 
                 *
                 * The new state and reward are not used, but are here to keep
                 * the interface uniform.
                 *
                 * @param s The previous state.
                 * @param s1 The new state.
                 * @param a The action performed.
                 * @param rew The reward obtained.
                 */
                virtual void updateSamplingQueue(size_t s, size_t s1, size_t a, double rew);

                /**
                 * @brief This function updates a QFunction based on simulated experience.
                 * 
                 * In DynaQ we sample N times from already experienced state-action pairs,
                 * and we update the resulting QFunction as if this experience was actually
                 * real.
                 *
                 * @param m The RLModel we sample experience from.
                 * @param q The QFunction to update.
                 */
                virtual void batchUpdateQ(const RLModel & m, QFunction * q);
            protected:
                // We use two structures because generally S*A is not THAT big, and we can definitely use
                // the O(1) insertion and O(1) sampling time.
                std::unordered_set<std::pair<size_t,size_t>, boost::hash<std::pair<size_t, size_t>>> visitedStatesActionsInserter_;
                std::vector<std::pair<size_t,size_t>> visitedStatesActionsSampler_;

                // Stuff for batch update
                mutable std::default_random_engine rand_;
        };
    }
}
#endif
