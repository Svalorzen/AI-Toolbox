#include <AIToolbox/Factored/MDP/Model.hpp>

#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {

    double Model::getTransitionProbability(const State & s, size_t a, const State & s1) const {
        return transitions_.makeDiffTransition(a).getTransitionProbability(S, s, s1);
    }

    double Model::getExpectedReward(const State & s, size_t a, const State &) const {
        return rewards_[a].getValue(S, s);
    }
}
