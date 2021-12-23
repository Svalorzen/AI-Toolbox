#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>

#include <AIToolbox/Impl/Seeder.hpp>

namespace AIToolbox::Factored::Bandit {

    LocalSearch::LocalSearch() : rnd_(Impl::Seeder::getSeed()) {}

    LocalSearch::Result LocalSearch::operator()(const Action & A, const Graph & graph) {
        Action startAction = makeRandomValue(A, rnd_);
        return (*this)(A, graph, std::move(startAction));
    }

    LocalSearch::Result LocalSearch::operator()(const Action & A, const Graph & graph, Action retAction) {
        double retval = 0.0;

        // Initialize ordering of agents
        std::iota(std::begin(agents_), std::end(agents_), 0);

        bool updated;
        do {
            updated = false;
            // Randomize order in which we iterate over agents to avoid
            // consistent collapse to local optimum for certain inputs.
            std::shuffle(std::begin(agents_), std::end(agents_), rnd_);

            for (auto a : agents_) {
                const auto & factors = graph.getFactors(a);

                size_t bestAction;
                double bestVal = std::numeric_limits<double>::lowest();

                // Find out which action of this agent is the locally best.
                for (size_t action = 0; action < A[a]; ++action) {
                    retAction[a] = action;
                    const double val = evaluateFactors(A, factors, retAction);

                    if (val > bestVal) {
                        updated = true;
                        bestVal = val;
                        bestAction = action;
                    }
                }
                retAction[a] = bestAction;
            }
            // Repeat until we were not able to improve a single agent.
        } while (updated);

        return {std::move(retAction), retval};
    }

    double LocalSearch::evaluateFactors(const Action & A, const Graph::FactorItList & factors, const Action & jointAction) const {
        double retval = 0.0;

        for (auto it : factors) {
            const auto & vars = it->getVariables();
            const auto & values = it->getData();

            retval += values[toIndexPartial(vars, A, jointAction)];
        }

        return retval;
    }
}
