#ifndef AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE
#define AI_TOOLBOX_FACTORED_MDP_COOPERATIVE_PRIORITIZED_SWEEPING_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/FasterTrie.hpp>
#include <AIToolbox/Factored/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Factored/MDP/Algorithms/Utils/CPSQueue.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class implements PrioritizedSweeping for cooperative environments.
     *
     * This class allows to perform prioritized sweeping in cooperative
     * environments.
     *
     * CooperativePrioritizedSweeping learns an approximation of the true
     * QFunction. After each interaction with the environment, the estimated
     * QFunction is updated. Additionally, a priority queue is updated which
     * keeps sets of the state and action spaces which are likely to need
     * updating.
     *
     * These sets are then sampled during batch updating, and the input model
     * (which should be also learned via environment interaction) is used to
     * sample new state-reward pairs to further refine the QFunction.
     *
     * @tparam M The type of the model to sample from.
     */
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

            CPSQueue queue_;

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
            queue_(model_.getS(), model_.getA(), model_.getTransitionFunction()),
            rand_(Impl::Seeder::getSeed())
    {
        const auto & ddn = model_.getTransitionFunction();

        // We weight the rewards so that they are split correctly between the
        // components of the QFunction.
        // Note that unused reward weights might result in r/0 or 0/0
        // operations, but since then we won't be using those elements
        // anyway it's not a problem.
        rewardWeights_.setZero();
        deltaStorage_.setZero();

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

    template <typename M>
    void CooperativePrioritizedSweeping<M>::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
        updateQ(s, a, s1, r.array() / rewardWeights_.array());
        addToQueue(s);
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::batchUpdateQ(const unsigned N) {
        // Initialize some variables to avoid reallocations
        State s(model_.getS().size());
        State s1(model_.getS().size());
        Action a(model_.getA().size());
        Rewards rews(model_.getS().size());

        for (size_t n = 0; n < N; ++n) {
            if (!queue_.getNonZeroPriorities()) return;

            queue_.reconstruct(s, a);

            // Filling randomly the missing elements.
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] == model_.getS()[i]) {
                    std::uniform_int_distribution<size_t> dist(0, model_.getS()[i]-1);
                    s[i] = dist(rand_);
                }
            }
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i] == model_.getA()[i]) {
                    std::uniform_int_distribution<size_t> dist(0, model_.getA()[i]-1);
                    a[i] = dist(rand_);
                }
            }

            // Finally, sample a new s1/rews from the model.
            model_.sampleSRs(s, a, &s1, &rews);
            rews.array() /= rewardWeights_.array();
            // And use them to update Q.
            updateQ(s, a, s1, rews);

            // Update the queue
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

            // Split the delta to each element referenced by this Q factor.
            // Note that we add to the storage, which is only cleared once we
            // call addToQueue; this means that multiple calls to this
            // functions cumulate their deltas.
            const auto delta = std::fabs(originalQ - q.values(sid, aid)) / q.tag.size();
            for (auto s : q.tag)
                deltaStorage_[s] += delta;
        }
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::addToQueue(const State & s1) {
        // Note that s1 was s before, but here we consider it as the
        // "future" state as we look for its parents.
        const auto & T = model_.getTransitionFunction();

        for (size_t i = 0; i < s1.size(); ++i) {
            // If the delta to apply is very small, we don't bother with it yet.
            // This allows us to save some work until it's actually worth it.
            if (deltaStorage_[i] < queue_.getNodeMaxPriority(i)) continue;
            const auto & aNode = T.nodes[i];
            for (size_t a = 0; a < aNode.nodes.size(); ++a) {
                const auto & sNode = aNode.nodes[a];
                for (size_t s = 0; s < static_cast<size_t>(sNode.matrix.rows()); ++s) {
                    // Compute the priority for this update (probability of
                    // transition times delta)
                    const auto p = sNode.matrix(s, s1[i]) * deltaStorage_[i];

                    // If it's not large enough, skip it.
                    if (p < theta_) continue;

                    queue_.update(i, a, s, p);
                }
            }
            // Reset this delta.
            deltaStorage_[i] = 0.0;
        }
    }
}

#endif
