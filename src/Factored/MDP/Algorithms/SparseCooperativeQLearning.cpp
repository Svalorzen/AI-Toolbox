#include <AIToolbox/Factored/MDP/Algorithms/SparseCooperativeQLearning.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox::Factored::MDP {
    SparseCooperativeQLearning::SparseCooperativeQLearning(State s, Action a, const double discount, const double alpha) :
            S(std::move(s)), A(std::move(a)), discount_(discount), alpha_(alpha), rules_(join(S, A)) {}

    void SparseCooperativeQLearning::reserveRules(const size_t s) {
        rules_.reserve(s);
    }

    void SparseCooperativeQLearning::insertRule(QFunctionRule rule) {
        auto factor = join(S.size(), rule.state, rule.action);
        rules_.emplace(factor, std::move(rule));
    }

    size_t SparseCooperativeQLearning::rulesSize() const {
        return rules_.size();
    }

    Action SparseCooperativeQLearning::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        Bandit::VariableElimination ve;

        const auto rules = rules_.filter(s1);
        const auto a1 = std::get<0>(ve(A, rules));

        auto beforeRules = rules_.filter(join(s, a));
        const auto afterRules = rules_.filter(join(s1, a1));

        std::vector<double> perAgentRews(A.size());
        // First, count how many before rules contain each agent.
        for (const auto & br : beforeRules)
            for (auto a : br.action.first)
                ++perAgentRews[a];
        // Then, weight the per-agent reward between the rules.
        for (size_t a = 0; a < A.size(); ++a)
            perAgentRews[a] = rew[a] / perAgentRews[a];
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
        for (auto & br : beforeRules) {
            double update = 0;
            for (auto a : br.action.first)
                update += perAgentRews[a];
            br.value += alpha_ * update;
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
