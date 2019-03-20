#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <boost/functional/hash.hpp>
#include <boost/heap/fibonacci_heap.hpp>

namespace AIToolbox::Factored::MDP {
    template <typename M>
    class CooperativePrioritizedSweeping {
        public:
            CooperativePrioritizedSweeping(const M & m, std::vector<std::vector<size_t>> basisDomains, double alpha = 0.3, double theta = 0.001) :
                    model_(m),
                    alpha_(alpha), theta_(theta),
                    qDomains_(std::move(basisDomains)),
                    rewardWeights_(model_.getS().size()),
                    gp_(model_.getS(), model_.getA(), q_),
                    ids_(join(model_.getS(), model_.getA())),
                    rand_(Impl::Seeder::getSeed())
            {
                const auto & ddn = model_.getTransitionFunction();

                // Note that unused reward weights might result in r/0 or 0/0
                // operations, but since then we won't be using those elements
                // anyway it's not a problem.
                rewardWeights_.setZero();

                q_.bases.reserve(qDomains_.size());
                for (const auto & domain : qDomains_) {
                    q_.bases.emplace_back();
                    auto & fm = q_.bases.back();

                    for (auto d : domain) {
                        // Note that there's one more Q factor that depends
                        // this state factor.
                        rewardWeights_[d] += 1.0;

                        // Compute state-action domain for this Q factor.
                        fm.actionTag = merge(fm.actionTag, ddn[d].actionTag);
                        for (const auto & n : ddn[d].nodes)
                            fm.tag = merge(fm.tag, n.tag);
                    }

                    // Initialize this factor's matrix.
                    const size_t sizeA = factorSpacePartial(fm.actionTag, model_.getA());
                    const size_t sizeS = factorSpacePartial(fm.tag, model_.getS());

                    fm.values.resize(sizeS, sizeA);
                    fm.values.setZero();
                }
            }

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
                    if (stateAction.first.size() == ids_.getFactors().size())
                        break;

                    ids_.refine(ids, stateAction);
                    if (!ids_.size())
                        break;
                }

                std::vector<size_t> missingS;
                std::vector<size_t> missingA;

                State s(model_.getS().size());
                Action a(model_.getA().size());

                // Copy stateAction values to s and a, and record missing ids.
                size_t x = 0;
                for (size_t i = 0; i < s.size(); ++i) {
                    if (x < stateAction.first.size() && i < stateAction.first[x])
                        missingS.push_back(i);

                    s[i] = stateAction.second[x++];
                }

                for (size_t i = 0; i < a.size(); ++i) {
                    if (x < stateAction.first.size() && i < stateAction.first[x])
                        missingA.push_back(i);

                    a[i] = stateAction.second[x++];
                }

                for (auto ss : missingS) {
                    std::uniform_int_distribution<size_t> dist(0, model_.getS()[ss]);
                    s[ss] = dist(rand_);
                }

                for (auto aa : missingA) {
                    std::uniform_int_distribution<size_t> dist(0, model_.getA()[aa]);
                    a[aa] = dist(rand_);
                }

                const auto [s1, r] = model_.sampleSRs(s, a);
                updateQ(s, a, s1, r.array() / rewardWeights_.array());

                // PartialFactorsEnumerator e(model_.getA(), missingA);
                // while (e.isValid()) {
                //     // Set missing actions.
                //     for (size_t i = 0; i < missingA.size(); ++i)
                //         a[e->first[i]] = a[e->second[i]];

                //     const auto [s1, r] = model_.sampleSR(s, a);
                //     updateQ(s, a, s1, r.array() / rewardWeights_.array());
                // }
            }

        private:
            std::vector<double> updateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                const auto a1 = gp_.sampleAction(s1);

                std::vector<double> deltasNoV(s.size());
                for (size_t i = 0; i < q_.bases.size(); ++i) {
                    auto & q = q_.bases[i];

                    const auto sid = toIndexPartial(q.tag, model_.getS(), s);
                    const auto aid = toIndexPartial(q.actionTag, model_.getA(), a);

                    const auto s1id = toIndexPartial(q.tag, model_.getS(), s1);
                    const auto a1id = toIndexPartial(q.actionTag, model_.getA(), a1);

                    auto delta = q.values(sid, aid);

                    double rr = 0.0;
                    for (auto s : qDomains_[i])
                        rr += r[s]; // already divided by weights

                    q.values(sid, aid) += alpha_ * ( rr + model_.getDiscount() * q.values(s1id, a1id) - q.values(sid, aid) );

                    delta = std::fabs(delta - q.values(sid, aid)) / q.tag.size();

                    for (auto s : q.tag)
                        deltasNoV[s] += delta;
                }
                return deltasNoV;
            }

            void addToQueue(const State & s1, const std::vector<double> & deltas) {
                // Note that s1 was s before, but here we consider it as the
                // "future" state as we look for its parents.

                const auto & T = model_.getTransitionFunction();

                for (size_t i = 0; i < s1.size(); ++i) {
                    const auto & aNode = T.nodes[i];
                    for (size_t a = 0; a < aNode.nodes.size(); ++a) {
                        const auto & sNode = aNode.nodes[a];
                        for (size_t s = 0; s < static_cast<size_t>(sNode.matrix.rows()); ++s) {
                            const auto p = sNode.matrix(s, s1[i]) * deltas[i];

                            if (p < theta_) continue;

                            Backup backup = PartialFactors{
                                join(sNode.tag, aNode.actionTag), // Keys
                                join(                             // Values
                                    toFactorsPartial(sNode.tag,       model_.getS(), s),
                                    toFactorsPartial(aNode.actionTag, model_.getA(), a)
                                )
                            };
                            auto hIt = findByBackup_.find(backup);

                            if (hIt != std::end(findByBackup_)) {
                                auto handle = hIt->second;

                                (*handle).priority += p;
                                queue_.increase(handle);
                            } else {
                                auto handle = queue_.emplace(PriorityQueueElement{p, backup});
                                auto id = ids_.insert(backup);

                                findById_[id] = handle;
                                findByBackup_[backup] = handle;
                            }
                        }
                    }
                }
            }

            const M & model_;

            double alpha_, theta_;

            std::vector<std::vector<size_t>> qDomains_;
            Vector rewardWeights_;

            FactoredMatrix2D q_;

            QGreedyPolicy gp_;

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
            std::unordered_map<size_t, typename QueueType::handle_type> findById_;
            std::unordered_map<Backup, typename QueueType::handle_type, boost::hash<Backup>> findByBackup_;

            mutable RandomEngine rand_;
    };
}

#endif
