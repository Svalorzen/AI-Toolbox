#ifndef AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE

#include <AIToolbox/MDP/DynaQInterface.hpp>

#include <queue>
#include <tuple>

#include <boost/heap/fibonacci_heap.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class represents the PrioritizedSweeping algorithm.
         */
        class PrioritizedSweeping : public DynaQInterface {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param alpha The learning rate of the QLearning method.
                 * @param discount The discount of the QLearning method.
                 * @param theta The queue threshold.
                 * @param n The number of sampling passes to do on the model upon batchUpdateQ().
                 */
                PrioritizedSweeping(size_t s, size_t a, double alpha = 0.5, double discount = 0.9, double theta = 0.5, unsigned n = 50);

                /**
                 * @brief This function updates the PrioritizedSweeping internal update queue.
                 * 
                 * Note that this function does NOT update the QFunction yet, but instead
                 * waits for the batchUpdateQ() call before doing that.
                 *
                 * @param s The previous state.
                 * @param s1 The new state.
                 * @param a The action performed.
                 * @param rew The reward obtained.
                 * @param q A pointer to the QFunction that is begin accessed.
                 */
                virtual void stepUpdateQ(size_t s, size_t s1, size_t a, double rew, const QFunction & q);

                /**
                 * @brief This function updates a QFunction based on simulated experience.
                 * 
                 * In PrioritizedSweeping we sample from the queue at most N times for 
                 * state action pairs that need updating. For each one of them we update
                 * the QFunction and recursively check whether this produces new changes
                 * worth updating. If so, they are inserted in the queue_ and the function
                 * proceeds to the nest iteration.
                 *
                 * @param m The RLModel we sample experience from.
                 * @param q The QFunction to update.
                 */
                virtual void batchUpdateQ(const RLModel & m, QFunction * q) override;

                size_t getQueueLength() const;
            private:
                double theta_;

                using PriorityQueueElement = std::tuple<double, size_t, size_t>;

                class PriorityTupleLess {
                    public:
                        bool operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const;
                };

                using QueueType = boost::heap::fibonacci_heap<PriorityQueueElement, boost::heap::compare<PriorityTupleLess>>;
                
                QueueType queue_;
                std::unordered_map<std::tuple<size_t, size_t>, QueueType::handle_type, boost::hash<std::tuple<size_t, size_t>>> queueHandles_;
        };
    }
}
#endif
