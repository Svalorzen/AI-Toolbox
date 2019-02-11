#include <AIToolbox/Factored/MDP/CooperativeExperience.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {

    CooperativeExperience::CooperativeExperience(State s, Action a, std::vector<FactoredDDN::Node> structure)
            : S(std::move(s)), A(std::move(a))
    {
        // init visits with unsigned structure
        rewards_.nodes = std::move(structure);

        for (size_t i = 0; i < S.size(); ++i) {
            for (size_t a = 0; a < rewards_.nodes.size(); ++a) {
                auto & rNode = rewards_.nodes[i].nodes[a];
                auto & vNode = visits_.nodes[i].nodes[a];

                const auto rows = factorSpacePartial(rNode.tag, S);

                rNode.matrix.resize(rows, S[i]+1);
                vNode.matrix.resize(rows, S[i]+1);
            }
        }

        indeces_.resize(S.size());
    }

    const CooperativeExperience::Indeces & CooperativeExperience::record(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        for (size_t ri = 0; ri < s1.size(); ++ri) {
            auto & vNode = visits_.nodes[ri];
            auto & rNode = rewards_.nodes[ri];

            // Compute action ID based on the actions that affect state factor 'i'.
            const auto actionId = toIndexPartial(vNode.actionTag, A, a);
            // Compute parent ID based on the parents of state factor 'i' under this action.
            const auto parentId = toIndexPartial(vNode.nodes[actionId].tag, S, s);

            // Update single values
            vNode.nodes[actionId].matrix(parentId, s1[ri]) += 1;
            rNode.nodes[actionId].matrix(parentId, s1[ri]) += rew[ri];
            // Update sums
            vNode.nodes[actionId].matrix(parentId, S[ri]) += 1;
            rNode.nodes[actionId].matrix(parentId, S[ri]) += rew[ri];

            // Save indeces to return to avoid recomputation.
            indeces_[ri] = {actionId, parentId};
        }
        return indeces_;
    }
}
