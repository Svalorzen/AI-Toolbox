#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>

#include <AIToolbox/Seeder.hpp>

namespace AIToolbox::Factored::Bandit {

    LocalSearch::LocalSearch() : rnd_(Seeder::getSeed()) {}

    LocalSearch::Result LocalSearch::operator()(const Action & A, const Graph & graph) {
        Action startAction = makeRandomValue(A, rnd_);
        return (*this)(A, graph, std::move(startAction));
    }

    LocalSearch::Result LocalSearch::operator()(const Action & A, const Graph & graph, Action retAction) {
        // Initialize ordering of agents
        agents_.resize(A.size());
        std::iota(std::begin(agents_), std::end(agents_), 0);

        bool updated;
        do {
            updated = false;
            // Randomize order in which we iterate over agents to avoid
            // consistent collapse to local optimum for certain inputs.
            std::shuffle(std::begin(agents_), std::end(agents_), rnd_);

            for (auto a : agents_) {
                const auto & factors = graph.getFactors(a);

                const size_t currentAction = retAction[a];
                double currentVal;

                size_t bestAction;
                double bestVal = std::numeric_limits<double>::lowest();

                // Find out which action of this agent is the locally best.
                for (size_t action = 0; action < A[a]; ++action) {
                    retAction[a] = action;
                    const double val = evaluateFactors(A, factors, retAction);

                    if (action == currentAction)
                        currentVal = val;

                    if (val > bestVal) {
                        bestVal = val;
                        bestAction = action;
                    }
                }
                // We only update if we increase the value; otherwise we leave
                // it as-is. This is to avoid updates which do not modify the
                // value of the action.
                if (bestVal > currentVal) {
                    retAction[a] = bestAction;
                    updated = true;
                } else {
                    retAction[a] = currentAction;
                }
            }
            // Repeat until we were not able to improve a single agent.
        } while (updated);

        // Compute actual value for the selected best action.
        const double retval = evaluateGraph(A, graph, retAction);

        return {std::move(retAction), retval};
    }

    double LocalSearch::evaluateGraph(const Action & A, const Graph & graph, const Action & jointAction) {
        double retval = 0.0;

        for (const auto & f : graph)
            retval += evaluateFactor(A, f, jointAction);

        return retval;
    }

    double LocalSearch::evaluateFactors(const Action & A, const Graph::FactorItList & factors, const Action & jointAction) {
        double retval = 0.0;

        for (auto it : factors)
            retval += evaluateFactor(A, *it, jointAction);

        return retval;
    }

    double LocalSearch::evaluateFactor(const Action & A, const Graph::FactorNode & factor, const Action & jointAction) {
        const auto & vars = factor.getVariables();
        const auto & values = factor.getData();

        return values[toIndexPartial(vars, A, jointAction)];
    }
}
