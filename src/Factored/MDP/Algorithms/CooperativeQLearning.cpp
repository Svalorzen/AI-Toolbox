#include <AIToolbox/Factored/MDP/Algorithms/CooperativeQLearning.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Factored/MDP/Utils.hpp>

namespace AIToolbox::Factored::MDP {
    CooperativeQLearning::CooperativeQLearning(const DDNGraph & g, const std::vector<std::vector<size_t>> & basisDomains, double discount, double alpha) :
            graph_(g), discount_(discount), alpha_(alpha),
            q_(makeQFunction(graph_, basisDomains)),
            policy_(graph_.getS(), graph_.getA(), q_),
            agentNormRews_(graph_.getA().size())
    {
        // We also pre-compute the perAgentRews_ here, since they do not depend on random subsets of rules.
        for (const auto & q : q_.bases)
            for (auto a : q.actionTag)
                ++agentNormRews_[a];
    }

    Action CooperativeQLearning::stepUpdateQ(const State & s, const Action & a, const State & s1, const Rewards & rew) {
        const auto a1 = policy_.sampleAction(s1);

        // We know in our case that we can only match a single "rule" in each q
        // basis. So we do the SparseCooperativeQLearning computations directly
        // on those.

        // Then, weight the per-agent reward between the rules.
        Vector perAgentRews(graph_.getA().size());
        perAgentRews.array() = rew.array() / agentNormRews_.array();

        // Now, for each after rule, add its weighted discounted value.
        for (const auto & q : q_.bases) {
            const auto s1id = toIndexPartial(q.tag, graph_.getS(), s1);
            const auto a1id = toIndexPartial(q.actionTag, graph_.getA(), a1);

            const double val = discount_ * q.values(s1id, a1id) / q.actionTag.size();
            for (auto a : q.actionTag)
                perAgentRews[a] += val;
        }

        // Finally, remove the weighted value of the original rules.
        for (const auto & q : q_.bases) {
            const auto sid = toIndexPartial(q.tag, graph_.getS(), s);
            const auto aid = toIndexPartial(q.actionTag, graph_.getA(), a);

            const double val = -q.values(sid, aid) / q.actionTag.size();
            for (auto a : q.actionTag)
                perAgentRews[a] += val;
        }

        // Update each rule weighted by the learning rate.
        perAgentRews.array() *= alpha_;

        for (auto & q : q_.bases) {
            const auto sid = toIndexPartial(q.tag, graph_.getS(), s);
            const auto aid = toIndexPartial(q.actionTag, graph_.getA(), a);

            double update = 0;
            for (auto a : q.actionTag)
                update += perAgentRews[a];
            q.values(sid, aid) += update;
        }

        return a1;
    }

    void CooperativeQLearning::setLearningRate(const double a) {
        if ( a <= 0.0 || a > 1.0 ) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
        alpha_ = a;
    }

    void CooperativeQLearning::setDiscount(const double d) {
        if ( d <= 0.0 || d > 1.0 ) throw std::invalid_argument("Discount parameter must be in (0,1]");
        discount_ = d;
    }

    void CooperativeQLearning::setQFunction(const double val) {
        for (auto & q : q_.bases)
            q.values.fill(val);
    }

    const DDNGraph &  CooperativeQLearning::getGraph() const { return graph_; }
    const State &  CooperativeQLearning::getS() const { return graph_.getS(); }
    const Action & CooperativeQLearning::getA() const { return graph_.getA(); }
    double CooperativeQLearning::getLearningRate() const { return alpha_; }
    double CooperativeQLearning::getDiscount() const { return discount_; }
    const FactoredMatrix2D & CooperativeQLearning::getQFunction() const { return q_; }
}
