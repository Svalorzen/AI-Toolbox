#include <AIToolbox/Factored/MDP/Algorithms/SparseCooperativeQLearning.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::MDP {
    FilterMap<QFunctionRule> initMap(const State & S, const Action & A, const std::vector<QFunctionRule> & rules) {
        FilterMap<QFunctionRule> map(join(S, A));

        map.reserve(rules.size());
        for (const auto & rule : rules) {
            auto factor = join(S.size(), rule.state, rule.action);
            map.emplace(factor, std::move(rule));
        }
        return map;
    }

    SparseCooperativeQLearning::SparseCooperativeQLearning(State s, Action a, const std::vector<QFunctionRule> & rules, const double discount, const double alpha) :
            S(std::move(s)), A(std::move(a)), discount_(discount), alpha_(alpha),
            rules_(initMap(S, A, rules)),
            policy_(S, A, rules_)
    {}

    Action SparseCooperativeQLearning::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        const auto a1 = policy_.sampleAction(s1);

        auto beforeRules = rules_.filter(join(s, a));
        const auto afterRules = rules_.filter(join(s1, a1));

        Vector perAgentRews(A.size());
        perAgentRews.setZero();

        // First, count how many before rules contain each agent.
        for (const auto & br : beforeRules)
            for (auto a : br.action.first)
                ++perAgentRews[a];

        // Then, weight the per-agent reward between the rules.
        perAgentRews.array() = rew.array() / perAgentRews.array();

        // Now, for each after rule, add its weighted discounted value.
        for (const auto & ar : afterRules) {
            const double val = discount_ * ar.value / ar.action.first.size();
            for (auto a : ar.action.first)
                perAgentRews[a] += val;
        }
        // Finally, remove the weighted value of the original rules.
        for (const auto & br : beforeRules) {
            const double val = -br.value / br.action.first.size();
            for (auto a : br.action.first)
                perAgentRews[a] += val;
        }
        // Update each rule weighted by the learning rate.
        perAgentRews.array() *= alpha_;

        for (auto & br : beforeRules) {
            double update = 0;
            for (auto a : br.action.first)
                update += perAgentRews[a];
            br.value += update;
        }

        return a1;
    }

    void SparseCooperativeQLearning::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    double SparseCooperativeQLearning::getLearningRate() const { return alpha_; }

    void SparseCooperativeQLearning::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    const State &  SparseCooperativeQLearning::getS() const { return S; }
    const Action & SparseCooperativeQLearning::getA() const { return A; }
    double SparseCooperativeQLearning::getDiscount() const { return discount_; }
    const FilterMap<QFunctionRule> & SparseCooperativeQLearning::getQFunctionRules() const { return rules_; }
}
