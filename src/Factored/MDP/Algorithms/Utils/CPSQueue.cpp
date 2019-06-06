#include <AIToolbox/Factored/MDP/Algorithms/Utils/CPSQueue.hpp>

#include <algorithm>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::Factored {
    CPSQueue::CPSQueue(State s, Action a, const FactoredDDN & ddn) :
            S(std::move(s)), A(std::move(a)),
            nonZeroPriorities_(0),
            order_(S.size()), nodes_(S.size()),
            rand_(Impl::Seeder::getSeed())
    {
        std::iota(std::begin(order_), std::end(order_), 0);
        for (size_t i = 0; i < ddn.nodes.size(); ++i) {
            const auto & ddnn = ddn.nodes[i];
            auto & n = nodes_[i];
            n.actionTag = ddnn.actionTag;
            n.maxV = 0.0;
            n.maxA = 0;
            n.nodes.resize(ddnn.nodes.size());
            n.order.resize(ddnn.nodes.size());
            std::iota(std::begin(n.order), std::end(n.order), 0);
            for (size_t j = 0; j < ddnn.nodes.size(); ++j) {
                auto & nn = n.nodes[j];
                nn.tag = ddnn.nodes[j].tag;
                nn.maxV = 0.0;
                nn.maxS = 0;
                nn.priorities.resize(ddnn.nodes[j].matrix.rows());
                nn.priorities.fill(-1.0);
            }
        }
    }

    void CPSQueue::update(size_t i, size_t a, size_t s, double p) {
        assert(p > 0.0);
        // If the priority was not assigned before, we increase the number of
        // non-zero "rules" in the queue.
        if (nodes_[i].nodes[a].priorities[s] <= 0.0)
            ++nonZeroPriorities_;
        // Note that we assume no duplicates so we don't need to look around.
        nodes_[i].nodes[a].priorities[s] += p;
        // We update the maxes if needed
        if (nodes_[i].nodes[a].priorities[s] > nodes_[i].nodes[a].maxV) {
            nodes_[i].nodes[a].maxV = nodes_[i].nodes[a].priorities[s];
            nodes_[i].nodes[a].maxS = s;

            if (nodes_[i].nodes[a].maxV > nodes_[i].maxV) {
                nodes_[i].maxV = nodes_[i].nodes[a].maxV;
                nodes_[i].maxA = a;
            }
        }
    }

    std::tuple<State, Action> CPSQueue::reconstruct() {
        // Initialize retval
        std::tuple<State, Action> retval;
        auto & [rets, reta] = retval;
        rets = S;
        reta = A;

        const auto partialMatch = [](const Factors & F, const Factors & f, const PartialKeys & keys, size_t id) {
            for (size_t i = 0; i < keys.size(); ++i) {
                const auto key = keys[i];
                const auto val = id % F[key];
                id /= F[key];
                if (f[key] == F[key] || f[key] == val) continue;
                return false;
            }
            return true;
        };

        const auto assignMatch = [](const Factors & F, Factors & f, const PartialKeys & keys, size_t id) {
            for (size_t i = 0; i < keys.size(); ++i) {
                const auto key = keys[i];
                f[key] = id % F[key];
                id /= F[key];
            }
        };

        // Random choice over nodes
        std::shuffle(std::begin(order_), std::end(order_), rand_);
        // Given action tag, random choice over compatible actions
        for (auto i : order_) {
            auto & node = nodes_[i];
            bool canTakeMax = partialMatch(A, reta, node.actionTag, node.maxA);
            canTakeMax = canTakeMax && partialMatch(S, rets, node.nodes[node.maxA].tag, node.nodes[node.maxA].maxS);
            if (canTakeMax) {
                // Add to reta and rets the new values
                assignMatch(A, reta, node.actionTag, node.maxA);
                assignMatch(S, rets, node.nodes[node.maxA].tag, node.nodes[node.maxA].maxS);

                // Update max of i
                auto & nn = node.nodes[node.maxA];
                if (nn.maxV > 0.0)
                    --nonZeroPriorities_;
                nn.priorities[nn.maxS] = 0.0;
                nn.maxV = nn.priorities.maxCoeff(&nn.maxS);

                node.maxA = 0;
                node.maxV = node.nodes[0].maxV;

                for (size_t a = 1; a < node.nodes.size(); ++a) {
                    if (node.nodes[a].maxV > node.maxV) {
                        node.maxA = a;
                        node.maxV = node.nodes[a].maxV;
                    }
                }
                continue;
            }
            // The default max is not compatible with our current set, so we
            // look around for a half-random, half best choice.
            //
            // Select random compatible action. We know that there is at least
            // one element in its node which is compatible with us (since it
            // contains all possible values of the parents).
            std::shuffle(std::begin(node.order), std::end(node.order), rand_);
            auto j = 0;
            for (auto jj : node.order) {
                if (partialMatch(A, reta, node.actionTag, jj)) {
                    assignMatch(A, reta, node.actionTag, jj);
                    j = jj;
                    break;
                }
            }

            // Select compatible parent set with highest priority
            auto && nn = node.nodes[j];
            auto x = nn.maxS;
            if (partialMatch(S, rets, nn.tag, nn.maxS)) {
                // If the current max is alright...
                // Set to zero
                if (nn.maxV > 0.0)
                    --nonZeroPriorities_;
                nn.priorities[nn.maxS] = 0.0;
                // Find the new max
                nn.maxV = nn.priorities.maxCoeff(&nn.maxS);
                // The new max is <= than the one we popped, and since we are
                // guaranteed not to have picked the global top pick (since
                // we're here looking in the first place) we don't have to
                // propagate the new max above.
            } else {
                // We have to find another one
                double ourMax = -2.0;

                for (size_t xx = 0; xx < static_cast<size_t>(nn.priorities.size()); ++xx) {
                    if (xx == nn.maxS) continue;
                    if (nn.priorities[xx] > ourMax && partialMatch(S, rets, nn.tag, xx)) {
                        ourMax = nn.priorities[xx];
                        x = xx;
                    }
                }
                // Set to zero
                if (ourMax > 0.0)
                    --nonZeroPriorities_;
                nn.priorities[x] = 0.0;
                // No max to update tho
            }
            assignMatch(S, rets, nn.tag, x);
        }
        return retval;
    }

    unsigned CPSQueue::getNonZeroPriorities() const {
        return nonZeroPriorities_;
    }
}
