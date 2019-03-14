#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>

namespace AIToolbox::Factored::MDP {
    template <typename M>
    class CooperativePrioritizedSweeping {
        public:
            void stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                auto delta1 = updateQ(s, a, s1, r);

                addToQueue(s, delta1);
            }

            void batchUpdateQ() {
                // Pick top element from queue
                //
                // Go over the whole queue (possibly in order), picking all
                // elements which are compatible with the previously picked
                // elements.
                //
                // Fill missing elements... randomly? Do them all?
                // - Action we can probably do all of them, and update v only once.
                //
                // Update s.
                // auto [_, i, s_i, a_i] = queue.pop;

                std::vector<size_t> missingA;

                PartialFactorsEnumerator e(A, missingA);
                State s;
                Action a;
                while (e.isValid()) {
                    // Set missing actions.
                    for (size_t i = 0; i < missingA.size(); ++i)
                        a[e->first[i]] = a[e->second[i]];

                    const auto [s1, r] = model_.sampleSR(s, a);
                    updateQ(s, a, s1, r);
                }
            }

        private:
            std::vector<double> updateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
                // Optional
                // const auto aa = gp_.sampleAction(s);

                const auto a1 = gp_.sampleAction(s1);

                std::vector<double> deltasNoV(s.size());
                for (size_t i = 0; i < q_.bases.size(); ++i) {
                    auto & q = q_[i];
                    const auto & v = v_[i];

                    const auto sid = toIndexPartial(q.tag, S, s);
                    const auto aid = toIndexPartial(q.actionTag, A, a);

                    const auto s1id = toIndexPartial(q.tag, S, s1);
                    const auto a1id = toIndexPartial(q.actionTag, A, a1);

                    auto delta = q.values(sid, aid);

                    q_(sid, aid) += alpha_ * ( r[i] + discount_ * q_(s1id, a1id) - q_(sid, aid) );

                    // Optional
                    // const auto aaid = toIndexPartial(q.actionTag, A, aa);
                    // delta = std::fabs(delta - q.values(sid, aaid)) / q.tag.size();

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
                                auto p = sNode.values(parentId, s[i]);
                                if (p * deltas[i] < theta_) continue;

                                //  queue.push(pv, {i, s_i}, {ps_i, pa_i})
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

            QGreedyPolicy gp_;
            FactoredMatrix2D q_;
            FactoredVector v_;
    };
}

#endif
