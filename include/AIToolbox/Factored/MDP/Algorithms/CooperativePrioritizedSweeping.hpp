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

            /**
             * @brief This function sets the QFunction to a set value.
             *
             * This function is useful to perform optimistic initialization.
             *
             * @param val The value to set all entries in the QFunction.
             */
            void setQFunction(double val);

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
            Vector rewardWeights_, deltaStorage_, rewardStorage_;

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
            rewardStorage_(model_.getS().size()),
            gp_(model_.getS(), model_.getA(), q_),
            queue_(model_.getGraph()),
            rand_(Impl::Seeder::getSeed())
    {
        // We weight the rewards so that they are split correctly between the
        // components of the QFunction.
        // Note that unused reward weights might result in r/0 or 0/0
        // operations, but since then we won't be using those elements
        // anyway it's not a problem.
        rewardWeights_.setZero();
        deltaStorage_.setZero();
        // We don't need to zero rewardStorage_

        const auto & nodes = model_.getGraph().getNodes();

        q_.bases.reserve(qDomains_.size());
        for (const auto & domain : qDomains_) {
            q_.bases.emplace_back();
            auto & q = q_.bases.back();

            for (const auto d : domain) {
                // Compute state-action domain for this Q factor.
                q.actionTag = merge(q.actionTag, nodes[d].agents);
                for (const auto & n : nodes[d].parents)
                    q.tag = merge(q.tag, n);
            }
            // We weight rewards based on the state features of each Q factor
            for (const auto d : q.tag)
                rewardWeights_[d] += 1.0;

            // Initialize this factor's matrix.
            const size_t sizeA = factorSpacePartial(q.actionTag, model_.getA());
            const size_t sizeS = factorSpacePartial(q.tag, model_.getS());

            q.values.resize(sizeS, sizeA);
            q.values.setZero();
        }
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
        updateQ(s, a, s1, r);
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

            // And use them to update Q.
            updateQ(s, a, s1, rews);

            // Update the queue
            addToQueue(s);
        }
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::updateQ(const State & s, const Action & a, const State & s1, const Rewards & r) {
        // Compute optimal action to do Q-Learning update.
        const auto a1 = gp_.sampleAction(s1);

        // The standard Q-update is in the form:
        //
        // Q(s,a) += alpha * ( R(s,a) + gamma * Q(s', a') - Q(s,a) )
        //
        // Since our Q-function is factored, we want to split the rewards per
        // state feature (similar to SparseCooperativeQLearning).

        // Start with R
        rewardStorage_ = r.array();
        // Now go over the factored Q-function for the rest
        for (const auto & q : q_.bases) {
            const auto sid = toIndexPartial(q.tag, model_.getS(), s);
            const auto aid = toIndexPartial(q.actionTag, model_.getA(), a);

            const auto s1id = toIndexPartial(q.tag, model_.getS(), s1);
            const auto a1id = toIndexPartial(q.actionTag, model_.getA(), a1);

            // gamma * Q(s', a') - Q(s, a)
            // We normalize it per state features, since we distribute the diff to all
            // elements of rewardStorage_.
            const auto diff = (model_.getDiscount() * q.values(s1id, a1id) - q.values(sid, aid)) / q.tag.size();

            // Apply the values to each state feature that applies to this Q factor.
            // R(s,a) + ...
            for (const auto s : q.tag)
                rewardStorage_[s] += diff;
        }

        // Normalize all values based on Q-factors
        rewardStorage_.array() /= rewardWeights_.array();
        rewardStorage_.array() *= alpha_;

        // We update each Q factor separately.
        for (size_t i = 0; i < q_.bases.size(); ++i) {
            auto & q = q_.bases[i];

            const auto sid = toIndexPartial(q.tag, model_.getS(), s);
            const auto aid = toIndexPartial(q.actionTag, model_.getA(), a);

            // Compute numerical reward from the components children of this Q
            // factor.
            double td = 0.0;
            for (const auto s : q.tag)
                td += rewardStorage_[s];

            q.values(sid, aid) += td;

            // Split the delta to each element referenced by this Q factor.
            // Note that we add to the storage, which is only cleared once we
            // call addToQueue; this means that multiple calls to this
            // functions cumulate their deltas.
            const auto delta = std::fabs(td) / q.tag.size();
            for (const auto s : q.tag)
                deltaStorage_[s] += delta;
        }
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::addToQueue(const State & s1) {
        // Note that s1 was s before, but here we consider it as the
        // "future" state as we look for its parents.
        const auto & T = model_.getTransitionFunction();
        const auto & graph = model_.getGraph();

        for (size_t i = 0; i < s1.size(); ++i) {
            // If the delta to apply is very small, we don't bother with it yet.
            // This allows us to save some work until it's actually worth it.
            if (deltaStorage_[i] < queue_.getNodeMaxPriority(i)) continue;

            // Here we need to iterate over j, but the queue still needs the
            // a,s variables. So we keep all of them in mind to keep things easy.
            size_t j = 0;
            for (size_t a = 0; a < graph.getPartialSize(i); ++a) {
                for (size_t s = 0; s < graph.getPartialSize(i, a); ++s) {
                    const auto p = T.transitions[i](j, s1[i]) * deltaStorage_[i];

                    // Increase j before we check if we want to skip.
                    ++j;
                    // If it's not large enough, skip it.
                    if (p < theta_) continue;

                    queue_.update(i, a, s, p);
                }
            }
            // Reset this delta.
            deltaStorage_[i] = 0.0;
        }
    }

    template <typename M>
    void CooperativePrioritizedSweeping<M>::setQFunction(const double val) {
        for (auto & q : q_.bases)
            q.values.fill(val);
    }

    template <typename M>
    const FactoredMatrix2D & CooperativePrioritizedSweeping<M>::getQFunction() const {
        return q_;
    }
}

#endif
