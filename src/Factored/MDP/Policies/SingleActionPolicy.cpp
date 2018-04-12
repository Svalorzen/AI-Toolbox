#include <AIToolbox/Factored/MDP/Policies/SingleActionPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    SingleActionPolicy::SingleActionPolicy(State s, Action a) :
            Base(std::move(s), std::move(a)), currentAction_(A.size()) {}

    Action SingleActionPolicy::sampleAction(const State &) const {
        return currentAction_;
    }

    double SingleActionPolicy::getActionProbability(const State &, const Action & a) const {
        return veccmp(a, currentAction_) == 0 ? 1.0 : 0.0;
    }

    void SingleActionPolicy::updateAction(Action a) {
        currentAction_ = std::move(a);
    }
}
