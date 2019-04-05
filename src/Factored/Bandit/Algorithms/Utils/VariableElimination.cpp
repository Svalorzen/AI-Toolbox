#include <AIToolbox/Factored/Bandit/Algorithms/Utils/VariableElimination.hpp>

#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>

namespace AIToolbox::Factored::Bandit {
    using VE = VariableElimination;

    namespace {
        struct Global {
            const Action & A;
            VE::Result result;

            size_t agent;
            VE::Factor newFactor;
            VE::Factor newCrossSum;

            void beginRemoval(size_t agent);
            void initNewFactor();
            void beginCrossSum(size_t agentAction);
            void crossSum(const VE::Factor & f);
            void endCrossSum();
            bool isValidNewFactor();
            void mergeFactors(VE::Factor & lhs, VE::Factor && rhs) const;
            void makeResult(VE::GVE::FinalFactors && finalFactors);
        };
    }

    VE::Result VE::operator()(const Action & A, GVE::Graph & graph) {
        GVE gve;
        Global global{A, {}, 0, {}, {}};

        gve(A, graph, global);

        return global.result;
    }

    void Global::beginRemoval(size_t currAgent) {
        // We save the currently eliminated agent to initialize the crossSum
        // tag correctly later.
        agent = currAgent;
    }

    void Global::initNewFactor() {
        // Here we only use this value as a marker to check that we have found
        // a max.
        newFactor.first = std::numeric_limits<double>::lowest();
    }

    void Global::beginCrossSum(size_t agentAction) {
        newCrossSum.first = 0.0;
        newCrossSum.second.clear();
        newCrossSum.second.emplace_back(agent, agentAction);
    }

    void Global::crossSum(const VE::Factor & factor) {
        // For each factor to sum, we add its value and we join tags with it.
        newCrossSum.first += factor.first;
        newCrossSum.second.insert(std::end(newCrossSum.second),
                std::begin(factor.second), std::end(factor.second));
    }

    void Global::endCrossSum() {
        // We only select the agent's best action.
        if (newCrossSum.first > newFactor.first) {
            newFactor.first = newCrossSum.first;
            std::swap(newFactor.second, newCrossSum.second);
        }
    }

    bool Global::isValidNewFactor() {
        // Simply check that we have found something at all. (maybe not even needed)
        return checkDifferentGeneral(newFactor.first, std::numeric_limits<double>::lowest());
    }

    void Global::mergeFactors(VE::Factor & lhs, VE::Factor && rhs) const {
        lhs.first += rhs.first;
        lhs.second.insert(std::end(lhs.second),
                std::begin(rhs.second), std::end(rhs.second));
    }

    void Global::makeResult(VE::GVE::FinalFactors && finalFactors) {
        result = std::make_tuple(Action(A.size()), 0.0);
        auto & [action, val] = result;

        for (const auto & f : finalFactors) {
            val += f.first;
            // Add tags together
            const auto & tags = f.second;
            for (size_t i = 0; i < tags.size(); ++i)
                action[tags[i].first] = tags[i].second;
        }
    }
}
