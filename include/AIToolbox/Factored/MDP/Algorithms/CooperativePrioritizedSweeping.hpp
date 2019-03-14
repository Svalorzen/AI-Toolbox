#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>

namespace AIToolbox::Factored::MDP {
    template <typename M>
    class CooperativePrioritizedSweeping {
        public:
            void stepUpdateQ(const State & s, const Action & a) {

                for (auto & q : q_.bases) {
                    const auto sid = toIndexPartial(q.tag, S, s);
                    const auto aid = toIndexPartial(q.actionTag, A, a);

                    q.values(sid, aid) = model_.getExpectedReward(s, a);

                    // Same as backpropagation, but just for this specific value.
                    //
                    // PartialFactorsEnumerator rDomain(S, v_i.tag);
                    // double futureVal = 0.0;
                    // for (size_t rId = 0; rDomain.isValid(); rDomain.advance(), ++rId)
                    //     futureVal += rhs.values[rId] * ddn.getTransitionProbability(space, actions, {q.tag, s}, {q.actionTag, a}, *rDomain);
                    // q.values(sid, aid) += model_.getDiscount() * futureVal;
                }

                // Update V
                double delta = 0.0;
                // auto a1 = greedyPolicy.sampleAction(s)
                // for (auto & v : v_) {
                //     const auto sid = toIndexPartial(v.tag, S, s);
                //     const auto qid = toIndexPartial(q.tag, S, s);
                //     const auto aid = toIndexPartial(q.actionTag, A, a1);
                //
                //     delta += v(sid) - q.values(qid, aid)
                //     v(sid) = q.values(qid, aid);
                // }
                delta = std::fabs(delta);

                // Distribute deltas to each s component
                std::vector<double> diffs(s.size());
                for (const auto & q : q_.bases)
                    for (auto s : q.tag)
                        diffs[s] += delta / q.tag.size();

                // Add elements to the queue
                for (size_t i = 0; i < s.size(); ++i) {
                    // for (ps_i : s_i)
                    //     for (pa_i : s_i)
                    //         auto p = model_.getTransitionProbability(ps_i, pa_i, s[i])
                    //         if (p * diffs[i] < theta) continue;
                    //         queue.push(pv, i, ps_i, pa_i)
                }
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

                // auto & q = q_[i];
                //
                // delta = 0.0
                // q.values(s_i, a_i) = model_.getExpectedReward(s_i, a_i);
                //
                // for (size_t i = 0; i < s_i.size(); ++i) {
                //     for (ps_i : s_i)
                //         for (pa_i : s_i)
                //             auto p = model_.getTransitionProbability(ps_i, pa_i, s[i])
                //             if (p * diffs[i] < theta) continue;
                //             queue.push(pv, i, ps_i, pa_i)
                // }
            }

            void stepUpdateQ(const State & s, Action & a, const std::vector<size_t> & missingA) {
                // PartialFactorsEnumerator enum(A, missingA);
                //
                // while (enum.isValid())
                {
                    // for (i : enum)
                    //     a[enum->first[i]] = a[enum->second[i]]

                    for (auto & q : q_.bases) {
                        const auto sid = toIndexPartial(q.tag, S, s);
                        const auto aid = toIndexPartial(q.actionTag, A, a);

                        q.values(sid, aid) = model_.getExpectedReward(s, a);

                        // Same as backpropagation, but just for this specific value.
                        //
                        // PartialFactorsEnumerator rDomain(S, v_i.tag);
                        // double futureVal = 0.0;
                        // for (size_t rId = 0; rDomain.isValid(); rDomain.advance(), ++rId)
                        //     futureVal += rhs.values[rId] * ddn.getTransitionProbability(space, actions, {q.tag, s}, {q.actionTag, a}, *rDomain);
                        // q.values(sid, aid) += model_.getDiscount() * futureVal;
                    }
                }

                // Update V

                // Work the deltas
            }

        private:
            const M & model_;

            State S;
            Action A;

            double discount_;

            FactoredMatrix2D q_;
    };
}

#endif
