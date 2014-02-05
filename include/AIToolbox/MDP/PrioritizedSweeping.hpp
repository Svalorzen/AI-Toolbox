#ifndef AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE
#define AI_TOOLBOX_MDP_PRIORITIZEDSWEEPING_HEADER_FILE

#include <tuple>
#include <unordered_map>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        class RLModel;
        /**
         * @brief This class represents the PrioritizedSweeping algorithm.
         */
        class PrioritizedSweeping {
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
                PrioritizedSweeping(const RLModel & m, double discount = 0.9, double theta = 0.5, unsigned n = 50);

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
                void stepUpdateQ(size_t s, size_t a);

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
                void batchUpdateQ();

                size_t getQueueLength() const;
                const RLModel & getModel() const;
                const QFunction & getQFunction() const;
                const ValueFunction & getValueFunction() const;

            private:
                size_t S, A;
                unsigned N;
                double discount_, theta_;

                const RLModel & model_;
                QFunction qfun_;
                ValueFunction vfun_;

                using PriorityQueueElement = std::tuple<double, size_t>;

                class PriorityTupleLess {
                    public:
                        bool operator() (const PriorityQueueElement& arg1, const PriorityQueueElement& arg2) const;
                };

                using QueueType = boost::heap::fibonacci_heap<PriorityQueueElement, boost::heap::compare<PriorityTupleLess>>;
                
                QueueType queue_;

                std::unordered_map<size_t, QueueType::handle_type> queueHandles_;
        };
    }
}
#endif
