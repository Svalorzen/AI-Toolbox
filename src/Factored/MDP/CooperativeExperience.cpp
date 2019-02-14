#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeExperience::CooperativeExperience(State s, Action a, std::vector<FactoredDDN::Node> structure)
            : S(std::move(s)), A(std::move(a))
    {
        // init visits with unsigned structure
        rewards_ = std::move(structure);
        visits_.resize(rewards_.size());

        for (size_t i = 0; i < S.size(); ++i) {
            visits_[i].reserve(rewards_[i].nodes.size());

            for (size_t a = 0; a < rewards_[i].nodes.size(); ++a) {
                auto & rNode = rewards_[i].nodes[a];

                const auto rows = factorSpacePartial(rNode.tag, S);

                rNode.matrix.resize(rows, S[i]+1);
                rNode.matrix.setZero();
                visits_[i].emplace_back(rows, S[i]+1);
                visits_[i].back().setZero();
            }
        }
        indeces_.resize(S.size());
    }

    const CooperativeExperience::Indeces & CooperativeExperience::record(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        for (size_t i = 0; i < S.size(); ++i) {
            auto & rNode = rewards_[i];

            // Compute action ID based on the actions that affect state factor 'i'.
            const auto actionId = toIndexPartial(rNode.actionTag, A, a);
            // Compute parent ID based on the parents of state factor 'i' under this action.
            const auto parentId = toIndexPartial(rNode.nodes[actionId].tag, S, s);

            // Update single values
            rNode.nodes[actionId].matrix(parentId, s1[i]) += rew[i];
            visits_[i][actionId](parentId, s1[i]) += 1;
            // Update sums
            rNode.nodes[actionId].matrix(parentId, S[i]) += rew[i];
            visits_[i][actionId](parentId, S[i]) += 1;

            // Save indeces to return to avoid recomputation.
            indeces_[i] = {actionId, parentId};
        }
        return indeces_;
    }

    void CooperativeExperience::reset() {
        for (size_t i = 0; i < S.size(); ++i) {
            for (size_t a = 0; a < rewards_[i].nodes.size(); ++a) {
                rewards_[i].nodes[a].matrix.setZero();
                visits_[i][a].setZero();
            }
        }
    }

    const CooperativeExperience::VisitTable & CooperativeExperience::getVisitTable() const {
        return visits_;
    }

    const CooperativeExperience::RewardMatrix & CooperativeExperience::getRewardMatrix() const {
        return rewards_;
    }

    const State & CooperativeExperience::getS() const { return S; }
    const Action & CooperativeExperience::getA() const { return A; }
}
