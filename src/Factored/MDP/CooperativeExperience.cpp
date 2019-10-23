#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeExperience::CooperativeExperience(const DDNGraph & graph)
            : graph_(graph)
    {
        const auto & S = graph_.getS();
        // init visits with unsigned structure
        rewards_.reserve(S.size());
        visits_.reserve(S.size());

        for (size_t i = 0; i < S.size(); ++i) {
            rewards_.emplace_back(graph_.getSize(i), S[i] + 1);
            rewards_.back().setZero();
            visits_.emplace_back(graph_.getSize(i), S[i] + 1);
            visits_.back().setZero();
        }
        indeces_.resize(S.size());
    }

    const CooperativeExperience::Indeces & CooperativeExperience::record(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        const auto & S = graph_.getS();
        for (size_t i = 0; i < S.size(); ++i) {
            auto & rNode = rewards_[i];
            auto & vNode = visits_[i];

            auto id = graph_.getId(i, s, a);

            // Update single values
            rNode(id, s1[i]) += rew[i];
            vNode(id, s1[i]) += 1;
            // Update sums
            rNode(id, S[i]) += rew[i];
            vNode(id, S[i]) += 1;

            // Save indeces to return to avoid recomputation.
            indeces_[i] = id;
        }
        return indeces_;
    }

    void CooperativeExperience::reset() {
        for (size_t i = 0; i < graph_.getS().size(); ++i) {
            rewards_[i].setZero();
            visits_[i].setZero();
        }
    }

    const CooperativeExperience::VisitTable & CooperativeExperience::getVisitTable() const {
        return visits_;
    }

    const CooperativeExperience::RewardMatrix & CooperativeExperience::getRewardMatrix() const {
        return rewards_;
    }

    const State & CooperativeExperience::getS() const { return graph_.getS(); }
    const Action & CooperativeExperience::getA() const { return graph_.getA(); }
    const DDNGraph & CooperativeExperience::getGraph() const { return graph_; }
}
