#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <boost/functional/hash.hpp>
#include <boost/heap/fibonacci_heap.hpp>

#include <iostream>

namespace AIToolbox::Factored::MDP {
    template <typename T>
    std::ostream & operator<<(std::ostream & os, const std::vector<T> & v) {
        for (auto vv : v)
            os << vv << ' ';
        return os;
    }
    std::ostream & operator<<(std::ostream & os, const PartialFactors & pf) {
        os << pf.first << " ==> " << pf.second;
        return os;
    }
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
                //std::cout << "Rewards weights: " << rewardWeights_.transpose() << '\n';
            }

            void stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                //std::cout << "Running new stepUpdateQ with:\n"
                //              "- s  = " << s  << '\n' <<
                //              "- a  = " << a  << '\n' <<
                //              "- s1 = " << s1 << '\n' <<
                //              "- r  = " << r.transpose()  << '\n';
                auto delta1 = updateQ(s, a, s1, r.array() / rewardWeights_.array());

                addToQueue(s, delta1);
            }

            void batchUpdateQ() {
                for (size_t n = 0; n < 50; ++n) {
                    if (queue_.empty()) return;

                    // Pick top element from queue
                    auto [p, id, stateAction] = queue_.top();
                    (void)p;

                    queue_.pop();

                    ids_.erase(id);
                    findById_.erase(id);
                    findByBackup_.erase(stateAction);

                    //std::cout << "BATCH UPDATE\n";
                    //std::cout << "Selected initial SA: " << stateAction << '\n';

                    // We want to remove as many rules in one swoop as possible, thus
                    // we take all rules compatible with our initial pick.
                    auto ids = ids_.filter(stateAction);
                    while (ids.size()) {
                        // Take a random compatible element to add to the first
                        // one. Ideally one would want to pick the one with the
                        // highest priority, but it's also very important to be as
                        // fast as possible here since we want to do as many
                        // updates as we can; thus, we do the easiest thing.
                        id = ids.back();
                        //std::cout << "Extracted additional index " << id << '\n';
                        ids.pop_back();

                        // Find the handle to the backup in the priority queue.
                        auto hIt = findById_.find(id);
                        assert(hIt != std::end(findById_));

                        auto handle = hIt->second;

                        //std::cout << "Additional SA: " << (*handle).stateAction << '\n';

                        // Add the selected state-action pair and add it to our
                        // own.
                        stateAction = merge(stateAction, (*handle).stateAction);
                        //std::cout << "Merged SA: " << stateAction << '\n';

                        // Remove the selected backup from all data-structures.
                        ids_.erase(id);
                        findById_.erase(hIt);
                        findByBackup_.erase((*handle).stateAction);
                        queue_.erase(handle);

                        // Refine with the remaining ids
                        ids = ids_.refine(ids, stateAction);
                    }

                    //std::cout << "Done merging: " << stateAction << "\n";

                    std::vector<size_t> missingS;
                    std::vector<size_t> missingA;

                    State s(model_.getS().size());
                    Action a(model_.getA().size());

                    // Copy stateAction values to s and a, and record missing ids.
                    size_t x = 0;
                    for (size_t i = 0; i < s.size(); ++i) {
                        if (x >= stateAction.first.size() || i < stateAction.first[x])
                            missingS.push_back(i);
                        else
                            s[i] = stateAction.second[x++];
                    }

                    //std::cout << "S: " << s << " ; missingS: " << missingS << " ; x = " << x << '\n';

                    for (size_t i = 0; i < a.size(); ++i) {
                        if (x >= stateAction.first.size() || i + model_.getS().size() < stateAction.first[x])
                            missingA.push_back(i);
                        else
                            a[i] = stateAction.second[x++];
                    }

                    //std::cout << "A: " << a << " ; missingA: " << missingA << '\n';

                    for (auto ss : missingS) {
                        std::uniform_int_distribution<size_t> dist(0, model_.getS()[ss]-1);
                        s[ss] = dist(rand_);
                    }

                    for (auto aa : missingA) {
                        std::uniform_int_distribution<size_t> dist(0, model_.getA()[aa]-1);
                        a[aa] = dist(rand_);
                    }

                    //std::cout << "Final S: " << s << " ; final A: " << a << '\n';

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
            }

            const FactoredMatrix2D & getQFunction() const {
                return q_;
            }

        private:
            std::vector<double> updateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                const auto a1 = gp_.sampleAction(s1);

                //std::cout << "Delta per Q component: ";

                std::vector<double> deltasNoV(s.size());
                for (size_t i = 0; i < q_.bases.size(); ++i) {
                    auto & q = q_.bases[i];

                    const auto sid = toIndexPartial(q.tag, model_.getS(), s);
                    const auto aid = toIndexPartial(q.actionTag, model_.getA(), a);

                    const auto s1id = toIndexPartial(q.tag, model_.getS(), s1);
                    const auto a1id = toIndexPartial(q.actionTag, model_.getA(), a1);


                    double rr = 0.0;
                    for (auto s : qDomains_[i])
                        rr += r[s]; // already divided by weights

                    auto delta = q.values(sid, aid);

                    q.values(sid, aid) += alpha_ * ( rr + model_.getDiscount() * q.values(s1id, a1id) - q.values(sid, aid) );

                    delta = std::fabs(delta - q.values(sid, aid)) / q.tag.size();
                    //std::cout << delta << ", ";

                    for (auto s : q.tag)
                        deltasNoV[s] += delta;
                }
                //std::cout << '\n';
                //std::cout << "Final deltas per-state: " << deltasNoV << '\n';
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
                                join(model_.getS().size(), sNode.tag, aNode.actionTag), // Keys
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
                                auto id = ids_.insert(backup);
                                auto handle = queue_.emplace(PriorityQueueElement{p, id, backup});

                                //std::cout << "Inserted in IDS [" << backup << "] with index " << id << '\n';
                                //std::cout << "    Value in queue: " << (*handle).stateAction << '\n';

                                findById_[id] = handle;
                                findByBackup_[backup] = handle;
                            }
                        }
                    }
                }

                //std::cout << "Queue now contains " << queue_.size() << " entries.\n";
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
                size_t id;
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
