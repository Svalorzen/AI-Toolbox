#include <AIToolbox/Factored/MDP/Model.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {

    double Model::getTransitionProbability(const State & s, size_t a, const State & s1) const {
        double p = 1.0;

        // For each partial transition matrix, we compute the entry which
        // applies to this transition, and we multiply all entries together.
        for (size_t i = 0; i < S.size(); ++i) {
            // Compute parent ID based on the parents of state factor 'i'
            auto parentId = toIndexPartial(transitions_[a][i].factors, S, s);
            p *= transitions_[a][i].matrix(parentId, s1[i]);
        }

        return p;
    }

    double Model::getExpectedReward(const State & s, size_t a, const State &) const {
        double rew = 0.0;

        // For each partial reward matrix, we sum all values that apply.
        for (size_t i = 0; i < rewards_.size(); ++i) {
            auto id = toIndexPartial(rewards_[i].factors, S, s);
            rew += rewards_[i].matrix(id, a);
        }

        return rew;
    }
}
