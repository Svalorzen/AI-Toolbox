#include <AIToolbox/Factored/Bandit/Policies/SingleActionPolicy.hpp>

#include <AIToolbox/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    SingleActionPolicy::SingleActionPolicy(Action a) :
            Base(std::move(a)), currentAction_(A.size()) {}

    Action SingleActionPolicy::sampleAction() const {
        return currentAction_;
    }

    double SingleActionPolicy::getActionProbability(const Action & a) const {
        return veccmp(a, currentAction_) == 0 ? 1.0 : 0.0;
    }

    void SingleActionPolicy::updateAction(Action a) {
        currentAction_ = std::move(a);
    }
}
