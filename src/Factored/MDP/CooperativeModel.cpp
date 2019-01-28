#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeModel::CooperativeModel(State s, Action a, FactoredDDN transitions, Factored2DMatrix rewards, const double discount) :
            S(std::move(s)), A(std::move(a)), discount_(discount),
            transitions_(std::move(transitions)), rewards_(std::move(rewards)),
            rand_(Impl::Seeder::getSeed()) {}

    std::tuple<State, double> CooperativeModel::sampleSR(const State & s, const Action & a) const {
        State s1(S.size());
        const double reward = sampleSR(s, a, &s1);

        return std::make_tuple(s1, reward);
    }

    double CooperativeModel::sampleSR(const State & s, const Action & a, State * s1p) const {
        State & s1 = *s1p;

        for (size_t i = 0; i < S.size(); ++i) {
            const auto actionId = toIndexPartial(transitions_[i].actionTag, A, a);

            const auto & node = transitions_[i].nodes[actionId];
            const auto parentId = toIndexPartial(node.tag, S, s);

            const size_t newS = sampleProbability(S[i], node.matrix.row(parentId), rand_);

            s1[i] = newS;
        }

        return rewards_.getValue(S, A, s, a);
    }

    double CooperativeModel::getTransitionProbability(const State & s, const Action & a, const State & s1) const {
        return transitions_.getTransitionProbability(S, A, s, a, s1);
    }

    double CooperativeModel::getExpectedReward(const State & s, const Action & a, const State &) const {
        return rewards_.getValue(S, A, s, a);
    }

    const State & CooperativeModel::getS() const { return S; }
    const Action & CooperativeModel::getA() const { return A; }
    double CooperativeModel::getDiscount() const { return discount_; }
    const FactoredDDN & CooperativeModel::getTransitionFunction() const { return transitions_; }
    const Factored2DMatrix & CooperativeModel::getRewardFunction() const { return rewards_; }
}

