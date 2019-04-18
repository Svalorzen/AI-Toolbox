#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/FasterTrie.hpp>
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
            /**
             * @brief Basic constructor.
             *
             * @param m The model to use for learning.
             * @param basisDomains The domains of the Q-Function to use.
             * @param alpha The alpha parameter of the Q-Learning update.
             * @param theta The threshold for queue inclusion.
             */
            CooperativePrioritizedSweeping(const M & m, std::vector<std::vector<size_t>> basisDomains, double alpha = 0.3, double theta = 0.001);

            /**
             * @brief This function performs a single update of the Q-Function with the input data.
             *
             * @param s The initial state.
             * @param a The action performed.
             * @param s1 The final state.
             * @param r The rewards obtained (one per state factor).
             */
            void stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r);

            /**
             * @brief This function performs a series of batch updates using the model to sample.
             *
             * The updates are generated from the contents of the queue, so
             * that the updates are done in priority order.
             *
             * @param N The number of priority updates to perform.
             */
            void batchUpdateQ(const unsigned N = 50);

            /**
             * @brief This function returns a reference to the internal QFunction.
             */
            const FactoredMatrix2D & getQFunction() const;

        private:
            /**
             * @brief This function performs the actual QFunction updates for both stepUpdateQ and batchUpdateQ.
             *
             * @param s The initial state.
             * @param a The action performed.
             * @param s1 The final state.
             * @param r The *normalized* rewards to use (one per state factor).
             */
            void updateQ(const State & s, const Action & a, const State & s1, const Rewards & r);

            /**
             * @brief This function updates the queue using the input state and the internal stored deltas.
             *
             * @param s1 The state to backpropagate deltas from.
             */
            void addToQueue(const State & s1);

            const M & model_;

            double alpha_, theta_;

            std::vector<std::vector<size_t>> qDomains_;
            Vector rewardWeights_, deltaStorage_;

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
            FasterTrie ids_;
            std::unordered_map<Backup, typename QueueType::handle_type, boost::hash<Backup>> findByBackup_;

            mutable RandomEngine rand_;
    };

    template <typename M>
    CooperativePrioritizedSweeping<M>::CooperativePrioritizedSweeping(const M & m, std::vector<std::vector<size_t>> basisDomains, double alpha, double theta) :
            model_(m),
            alpha_(alpha), theta_(theta),
            qDomains_(std::move(basisDomains)),
            rewardWeights_(model_.getS().size()),
            deltaStorage_(model_.getS().size()),
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

    template <typename M>
    void CooperativePrioritizedSweeping<M>::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
        //std::cout << "Running new stepUpdateQ with:\n"
        //              "- s  = " << s  << '\n' <<
        //              "- a  = " << a  << '\n' <<
        //              "- s1 = " << s1 << '\n' <<
        //              "- r  = " << r.transpose()  << '\n';
        updateQ(s, a, s1, r.array() / rewardWeights_.array());
        addToQueue(s);
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::batchUpdateQ(const unsigned N) {
        for (size_t n = 0; n < N; ++n) {
            if (queue_.empty()) return;

            // Pick top element from queue
            auto [priority, stateAction] = queue_.top();

            auto [ids, factor] = ids_.reconstruct(stateAction, true);

            for (const auto & id : ids) {
                auto hIt = findByBackup_.find(id.second);

                auto handle = hIt->second;

                queue_.erase(handle);
                findByBackup_.erase(hIt);
            }

            //std::cout << "Done merging: " << stateAction << "\n";

            State s(model_.getS().size());
            Action a(model_.getA().size());

            // Copy stateAction values to s and a, and record missing ids.
            size_t x = 0;
            for (size_t i = 0; i < s.size(); ++i, ++x) {
                if (factor[x] == model_.getS()[i]) {
                    std::uniform_int_distribution<size_t> dist(0, model_.getS()[i]-1);
                    s[i] = dist(rand_);
                } else {
                    s[i] = factor[x];
                }
            }

            for (size_t i = 0; i < a.size(); ++i, ++x) {
                if (factor[x] == model_.getA()[i]) {
                    std::uniform_int_distribution<size_t> dist(0, model_.getA()[i]-1);
                    a[i] = dist(rand_);
                } else {
                    a[i] = factor[x];
                }
            }

            //std::cout << "Final S: " << s << " ; final A: " << a << '\n';

            const auto [s1, r] = model_.sampleSRs(s, a);

            updateQ(s, a, s1, r.array() / rewardWeights_.array());
            // Since adding to queue is a relatively expensive operation, we
            // only update it once in a while. Here we update it if the
            // priority of the max element we have just popped off the queue is
            // lower than the current max update.
            //
            // If this is not called, each updateQ accumulates its changes to
            // the deltaStorage_.
            if (deltaStorage_.maxCoeff() > priority)
                addToQueue(s);
        }
    }

    template <typename M>
    const FactoredMatrix2D & CooperativePrioritizedSweeping<M>::getQFunction() const {
        return q_;
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::updateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
        // Compute optimal action to do Q-Learning update.
        const auto a1 = gp_.sampleAction(s1);

        // We update each Q factor separately.
        for (size_t i = 0; i < q_.bases.size(); ++i) {
            auto & q = q_.bases[i];

            const auto sid = toIndexPartial(q.tag, model_.getS(), s);
            const auto aid = toIndexPartial(q.actionTag, model_.getA(), a);

            const auto s1id = toIndexPartial(q.tag, model_.getS(), s1);
            const auto a1id = toIndexPartial(q.actionTag, model_.getA(), a1);

            // Compute numerical reward from the components children of this Q
            // factor.
            double rr = 0.0;
            for (auto s : qDomains_[i])
                rr += r[s]; // already divided by weights

            const auto originalQ = q.values(sid, aid);

            // Q-Learning update
            q.values(sid, aid) += alpha_ * ( rr + model_.getDiscount() * q.values(s1id, a1id) - q.values(sid, aid) );

            //std::cout << (std::fabs(originalQ - q.values(sid, aid)) / q.tag.size()) << ", ";

            // Split the delta to each element referenced by this Q factor.
            // Note that we add to the storage, which is only cleared once we
            // call addToQueue; this means that multiple calls to this
            // functions cumulate their deltas.
            for (auto s : q.tag)
                deltaStorage_[s] += std::fabs(originalQ - q.values(sid, aid)) / q.tag.size();
;
        }
        //std::cout << '\n';
        //std::cout << "Final deltas per-state: " << deltasNoV << '\n';
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::addToQueue(const State & s1) {
        // Note that s1 was s before, but here we consider it as the
        // "future" state as we look for its parents.
        const auto & T = model_.getTransitionFunction();

        for (size_t i = 0; i < s1.size(); ++i) {
            const auto & aNode = T.nodes[i];
            for (size_t a = 0; a < aNode.nodes.size(); ++a) {
                const auto & sNode = aNode.nodes[a];

                // We pre-allocate the Backup node (since it's size it's going
                // to remain the same throughout these loops. We'll change the
                // 'state' part in the loop.
                Backup backup{
                    join(model_.getS().size(), sNode.tag, aNode.actionTag), // Keys
                    {}
                };
                backup.second.resize(backup.first.size());
                // Write the values for this action in, since they won't change.
                toFactorsPartial(backup.second.begin() + sNode.tag.size(), aNode.actionTag, model_.getA(), a);

                for (size_t s = 0; s < static_cast<size_t>(sNode.matrix.rows()); ++s) {
                    // Compute the priority for this update (probability of
                    // transition times delta)
                    const auto p = sNode.matrix(s, s1[i]) * deltaStorage_[i];

                    // If it's not large enough, skip it.
                    if (p < theta_) continue;

                    // Write the values for this state.
                    toFactorsPartial(backup.second.begin(), sNode.tag, model_.getS(), s);

                    auto hIt = findByBackup_.find(backup);

                    if (hIt != std::end(findByBackup_)) {
                        // If we already had this entry, increase its priority.
                        auto handle = hIt->second;

                        (*handle).priority += p;
                        queue_.increase(handle);
                    } else {
                        // Otherwise create a new entry in the queue.
                        auto handle = queue_.emplace(PriorityQueueElement{p, backup});

                        //std::cout << "Inserted in IDS [" << backup << "] with index " << id << '\n';
                        //std::cout << "    Value in queue: " << (*handle).stateAction << '\n';

                        findByBackup_[backup] = handle;
                        ids_.insert(backup);
                    }
                }
            }
        }
        // Reset all deltas since we have updated the queue from them.
        deltaStorage_.setZero();

        //std::cout << "Queue now contains " << queue_.size() << " entries.\n";
    }
}

#endif
