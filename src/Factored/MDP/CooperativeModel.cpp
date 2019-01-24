#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

namespace AIToolbox::Factored::MDP {
    const State & CooperativeModel::getS() const { return S; }
    const Action & CooperativeModel::getA() const { return A; }
    double CooperativeModel::getDiscount() const { return discount_; }
    const FactoredDDN & CooperativeModel::getTransitionFunction() const { return transitions_; }
    const Factored2DMatrix & CooperativeModel::getRewardFunction() const { return rewards_; }
}
