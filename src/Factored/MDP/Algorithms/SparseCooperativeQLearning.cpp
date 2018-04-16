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
        Bandit::VariableElimination ve(A);

        const auto rules = rules_.filter(s1, 0); // Partial filter using only s1
        const auto a1 = std::get<0>(ve(rules));

        auto beforeRules = rules_.filter(join(s, a));
        const auto afterRules = rules_.filter(join(s1, a1));

        const auto computeQ = [](const size_t agent, const decltype(rules_)::Iterable & rules) {
            double sum = 0.0;
            for (const auto & rule : rules)
                sum += sequential_sorted_contains(rule.action.first, agent) ? rule.value / rule.action.first.size() : 0.0;
            return sum;
        };
        // First we compute all updates since we don't want to risk
        // overwriting the rules before we are done.
        std::vector<double> updates;
        updates.reserve(beforeRules.size());
        for (const auto & br : beforeRules) {
            double sum = 0;
            for (const auto agent : br.action.first) {
                sum += rew[agent];
                sum += discount_ * computeQ(agent, afterRules);
                sum -= computeQ(agent, beforeRules);
            }
            updates.push_back(alpha_ * sum);
        }
        // Finally update the rules.
        size_t i = 0;
        for (auto & br : beforeRules)
            br.value += updates[i++];

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
    const FactoredContainer<QFunctionRule> & SparseCooperativeQLearning::getQFunctionRules() const { return rules_; }
}
