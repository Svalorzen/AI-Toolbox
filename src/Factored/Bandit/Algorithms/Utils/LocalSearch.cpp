#include <AIToolbox/Factored/Bandit/Algorithms/Utils/LocalSearch.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <iostream>

std::ostream & operator<<(std::ostream & os, const std::vector<size_t> & out) {
    std::cout << '[';
    for (auto o : out) std::cout << o << ' ';
    std::cout << ']';
    return os;
}

namespace AIToolbox::Factored::Bandit {

    LocalSearch::LocalSearch() : rnd_(Impl::Seeder::getSeed()) {}

    LocalSearch::Result LocalSearch::operator()(const Action & A, const Graph & graph) {
        Action startAction = makeRandomValue(A, rnd_);
        return (*this)(A, graph, std::move(startAction));
    }

    LocalSearch::Result LocalSearch::operator()(const Action & A, const Graph & graph, Action retAction) {
        std::cout << "StartAction = " << retAction << '\n';

        // Initialize ordering of agents
        agents_.resize(A.size());
        std::iota(std::begin(agents_), std::end(agents_), 0);

        std::cout << "Agent setup: " << agents_ << '\n';

        bool updated;
        do {
            std::cout << "Step..., current retact = " << retAction << "\n";
            updated = false;
            // Randomize order in which we iterate over agents to avoid
            // consistent collapse to local optimum for certain inputs.
            std::shuffle(std::begin(agents_), std::end(agents_), rnd_);
            std::cout << "Agents will be processed in the following order: " << agents_ << '\n';

            for (auto a : agents_) {
                const auto & factors = graph.getFactors(a);

                const size_t currentAction = retAction[a];
                size_t bestAction;
                double bestVal = std::numeric_limits<double>::lowest();
                std::cout << "Processing agent " << a << "... current action = " << currentAction << '\n';

                // Find out which action of this agent is the locally best.
                for (size_t action = 0; action < A[a]; ++action) {
                    retAction[a] = action;
                    const double val = evaluateFactors(A, factors, retAction);

                    if (val > bestVal) {
                        bestVal = val;
                        bestAction = action;
                    }
                }
                retAction[a] = bestAction;
                updated |= (bestAction != currentAction);
                std::cout << "Updating action for agent " << a << ": " << currentAction << " --> " << bestAction << '\n';
            }
            // Repeat until we were not able to improve a single agent.
        } while (updated);

        // Compute actual value for the selected best action.
        const double retval = evaluateGraph(A, graph, retAction);

        return {std::move(retAction), retval};
    }

    double LocalSearch::evaluateGraph(const Action & A, const Graph & graph, const Action & jointAction) const {
        double retval = 0.0;

        for (const auto & f : graph)
            retval += evaluateFactor(A, f, jointAction);

        return retval;
    }

    double LocalSearch::evaluateFactors(const Action & A, const Graph::FactorItList & factors, const Action & jointAction) const {
        double retval = 0.0;

        for (auto it : factors)
            retval += evaluateFactor(A, *it, jointAction);

        return retval;
    }

    double LocalSearch::evaluateFactor(const Action & A, const Graph::FactorNode & factor, const Action & jointAction) const {
        const auto & vars = factor.getVariables();
        const auto & values = factor.getData();

        return values[toIndexPartial(vars, A, jointAction)];
    }
}
