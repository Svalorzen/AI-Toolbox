#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

#include <boost/heap/fibonacci_heap.hpp>

namespace AIToolbox::Factored::MDP {
    template <typename M>
    class CooperativePrioritizedSweeping {
        public:
            void stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                auto delta1 = updateQ(s, a, s1, r.array() / rewardWeights_.array());

                addToQueue(s, delta1);
            }

            void batchUpdateQ() {
                // Pick top element from queue
                auto [p, stateAction] = queue_.pop();
                (void)p;

                auto ids = ids_.filter(stateAction);
                while (true) {
                    // Take a random compatible element to add to the first
                    // one. Ideally one would want to pick the one with the
                    // highest priority, but it's also very important to be as
                    // fast as possible here since we want to do as many
                    // updates as we can; thus, we do the easiest thing.
                    auto id = ids.pop_back();

                    // Find the handle to the backup in the priority queue.
                    auto hIt = findById_.find(id);
                    auto handle = hIt->second;

                    // Add the selected state-action pair and add it to our
                    // own.
                    stateAction = merge(stateAction, (*handle).stateAction);

                    // Remove the selected backup from all data-structures.
                    ids_.erase(id);
                    findById_.remove(hIt);
                    findByBackup_.erase((*handle).stateAction);
                    queue_.erase(handle);

                    // If we have completed the state-action pair, we are done.
                    if (stateAction.first.size() == S.size() + A.size())
                        break;

                    ids_.refine(ids, stateAction);
                    if (!ids_.size())
                        break;
                }

                State s(S.size());
                Action a(A.size());

                // Determine missing S
                // Fix them randomly

                // Determine missin A
                // Put ids here
                std::vector<size_t> missingA;

                PartialFactorsEnumerator e(A, missingA);
                while (e.isValid()) {
                    // Set missing actions.
                    for (size_t i = 0; i < missingA.size(); ++i)
                        a[e->first[i]] = a[e->second[i]];

                    const auto [s1, r] = model_.sampleSR(s, a);
                    updateQ(s, a, s1, r.array() / rewardWeights_.array());
                }
            }

        private:
            std::vector<double> updateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                const auto a1 = gp_.sampleAction(s1);

                std::vector<double> deltasNoV(s.size());
                for (size_t i = 0; i < q_.bases.size(); ++i) {
                    auto & q = q_[i];

                    const auto sid = toIndexPartial(q.tag, S, s);
                    const auto aid = toIndexPartial(q.actionTag, A, a);

                    const auto s1id = toIndexPartial(q.tag, S, s1);
                    const auto a1id = toIndexPartial(q.actionTag, A, a1);

                    auto delta = q.values(sid, aid);

                    double rr = 0.0;
                    for (auto s : q.tag)
                        rr += r[s]; // already divided by weights

                    q_(sid, aid) += alpha_ * ( rr + discount_ * q_(s1id, a1id) - q_(sid, aid) );

                    delta = std::fabs(delta - q.values(sid, aid)) / q.tag.size();

                    for (auto s : q.tag)
                        deltasNoV[s] += delta;
                }
                return deltasNoV;
            }

            void addToQueue(const State & s, const std::vector<double> & deltas) {
                // Add elements to the queue
                const auto & T = model_.getTransitionFunction();

                for (size_t i = 0; i < s.size(); ++i) {
                    for (const auto & aNode : T[i].nodes) {
                        for (const auto & sNode : aNode.nodes) {
                            for (size_t parentId = 0; i < sNode.values.rows(); ++parentId) {
                                const auto p = sNode.values(parentId, s[i]) * deltas[i];

                                if (p < theta_) continue;

                                Backup backup;
                                auto hIt = findByBackup_.find(backup);

                                if (hIt != std::end(findByBackup_)) {
                                    auto handle = hIt->second;

                                    (*handle).priority += p;
                                    queue_.increase(handle);
                                } else {
                                    auto handle = queue_.push(p, backup);
                                    auto id = ids_.insert(backup);

                                    findById_[id] = handle;
                                    findByBackup_[backup] = handle;
                                }
                            }
                        }
                    }
                }
            }

            const M & model_;

            State S;
            Action A;

            double discount_, alpha_;
            double theta_;
            Vector rewardWeights_;

            QGreedyPolicy gp_;
            FactoredMatrix2D q_;

            using Backup = PartialFactors;

            struct PriorityQueueElement {
                double priority;
                Backup stateAction;
                bool operator<(const PriorityQueueElement& arg2) const {
                    return priority < arg2.priority;
                }
            };

            using QueueType = boost::heap::fibonacci_heap<PriorityQueueElement>;

            QueueType queue_;
            Trie ids_;
            std::unordered_map<size_t, typename QueueType::handle> findById_;
            std::unordered_map<Backup, typename QueueType::handle> findByBackup_;
    };
}

#endif
