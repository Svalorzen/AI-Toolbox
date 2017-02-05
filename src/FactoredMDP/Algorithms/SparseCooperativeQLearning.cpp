#include <AIToolbox/FactoredMDP/Algorithms/SparseCooperativeQLearning.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/FactoredMDP/Utils.hpp>
#include <AIToolbox/FactoredMDP/Algorithms/Utils/VariableElimination.hpp>

namespace AIToolbox {
    namespace FactoredMDP {
        SparseCooperativeQLearning::SparseCooperativeQLearning(State s, Action a, double discount, double alpha) :
                S(s), A(a), discount_(discount), alpha_(alpha), rules_(join(S, A)) {}

        void SparseCooperativeQLearning::reserveRules(size_t s) {
            rules_.reserve(s);
        }

        void SparseCooperativeQLearning::insertRule(QFunctionRule rule) {
            rules_.emplace(join(S.size(), rule.s_, rule.a_), std::move(rule));
        }

        size_t SparseCooperativeQLearning::rulesSize() const {
            return rules_.size();
        }

        Action SparseCooperativeQLearning::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & rew) {
            VariableElimination ve(A);

            auto rules = rules_.filter(s1, 0); // Partial filter using only s1
            auto a1 = ve(rules);

            auto beforeRules = rules_.filter(join(s, a));
            auto afterRules = rules_.filter(join(s1, std::get<0>(a1)));

            auto computeQ = [](size_t agent, const decltype(rules_)::Iterable & rules) {
                double sum = 0.0;
                for (const auto & rule : rules)
                    sum += sequential_sorted_contains(rule.a_.first, agent) ? rule.value_ / rule.a_.first.size() : 0.0;
                return sum;
            };
            // First we compute all updates since we don't want to risk
            // overwriting the rules before we are done.
            std::vector<double> updates;
            updates.reserve(beforeRules.size());
            for (const auto & br : beforeRules) {
                double sum = 0;
                for (const auto agent : br.a_.first) {
                    sum += rew[agent];
                    sum += discount_ * computeQ(agent, afterRules);
                    sum -= computeQ(agent, beforeRules);
                }
                updates.push_back(alpha_ * sum);
            }
            // Finally update the rules.
            size_t i = 0;
            for (auto & br : beforeRules)
                br.value_ += updates[i++];

            return std::get<0>(a1);
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

        double SparseCooperativeQLearning::getDiscount() const { return discount_; }

        const State &  SparseCooperativeQLearning::getS() const { return S; }
        const Action & SparseCooperativeQLearning::getA() const { return A; }
        const FactoredContainer<QFunctionRule> & SparseCooperativeQLearning::getQFunctionRules() const { return rules_; }
    }
}
