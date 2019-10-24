#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeExperience::CooperativeExperience(const DDNGraph & graph)
            : graph_(graph), timesteps_(0)
    {
        const auto & S = graph_.getS();
        // init visits with unsigned structure
        rewards_.reserve(S.size());
        visits_.reserve(S.size());

        for (size_t i = 0; i < S.size(); ++i) {
            rewards_.emplace_back(graph_.getSize(i));
            rewards_.back().setZero();
            M2s_.emplace_back(graph_.getSize(i));
            M2s_.back().setZero();
            visits_.emplace_back(graph_.getSize(i), S[i] + 1);
            visits_.back().setZero();
        }
        indeces_.resize(S.size());
    }

    const CooperativeExperience::Indeces & CooperativeExperience::record(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        ++timesteps_;

        const auto & S = graph_.getS();
        for (size_t i = 0; i < S.size(); ++i) {
            auto & rNode = rewards_[i];
            auto & mNode = M2s_[i];
            auto & vNode = visits_[i];

            auto id = graph_.getId(i, s, a);

            // Count updates
            vNode(id, s1[i]) += 1; // Single
            vNode(id, S[i]) += 1;  // Sum

            const auto delta = rew[i] - rNode(id);
            // Rolling average for this s,a,s1 tuple
            rNode(id) += delta / vNode(id, S[i]);
            // Rolling sum of square diffs.
            mNode(id) += delta * (rew[i] - rNode(id));

            // Save indeces to return to avoid recomputation.
            indeces_[i] = id;
        }
        return indeces_;
    }

    void CooperativeExperience::reset() {
        for (size_t i = 0; i < graph_.getS().size(); ++i) {
            rewards_[i].setZero();
            M2s_[i].setZero();
            visits_[i].setZero();
        }
        timesteps_ = 0;
    }

    unsigned long CooperativeExperience::getTimesteps() const {
        return timesteps_;
    }

    const CooperativeExperience::VisitsTable & CooperativeExperience::getVisitsTable() const {
        return visits_;
    }

    const CooperativeExperience::RewardMatrix & CooperativeExperience::getRewardMatrix() const {
        return rewards_;
    }

    const CooperativeExperience::RewardMatrix & CooperativeExperience::getM2Matrix() const {
        return M2s_;
    }

    const State & CooperativeExperience::getS() const { return graph_.getS(); }
    const Action & CooperativeExperience::getA() const { return graph_.getA(); }
    const DDNGraph & CooperativeExperience::getGraph() const { return graph_; }
}
