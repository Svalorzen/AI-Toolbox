#include <AIToolbox/Factored/Bandit/Algorithms/Utils/ReusingIterativeLocalSearch.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Factored::Bandit {
    ReusingIterativeLocalSearch::ReusingIterativeLocalSearch(const double resetActionProbability, const double randomizeFactorProbability, const unsigned trialNum) :
        resetActionProbability_(resetActionProbability),
        randomizeFactorProbability_(randomizeFactorProbability),
        trialNum_(trialNum),
        value_(std::numeric_limits<double>::lowest()),
        rnd_(Impl::Seeder::getSeed())
    {}

    ReusingIterativeLocalSearch::Result ReusingIterativeLocalSearch::operator()(const Action & A, const Graph & graph, const bool resetAction) {
        // If we haven't initialized the action yet, do so.
        // Otherwise we keep the same, hoping that the graph has not
        // changed too much and that the new optimum is close to our
        // old one.
        //
        // If needed we can force the reset though (in case we know
        // that the new graph has nothing in common with the old one).
        if (resetAction || value_ == std::numeric_limits<double>::lowest())
            std::tie(action_, value_) = ls_(A, graph);

        // In the trials we look around to see if we got stuck in some
        // local optima.
        for (unsigned i = 0; i < trialNum_; ++i) {
            // Either we completely start again from scratch...
            if (probabilityDistribution(rnd_) < resetActionProbability_) {
                newAction_ = makeRandomValue(A, rnd_);
            } else {
                newAction_ = action_;

                // Or we start from our current best action, and
                // randomize groups of agents. Note that we do not
                // randomize one agent at a time because then it
                // becomes much more likely that LS will simply climb
                // back to the same action as before.
                for (const auto & f : graph) {
                    if (probabilityDistribution(rnd_) >= randomizeFactorProbability_)
                        continue;

                    for (auto a : f.getVariables()) {
                        std::uniform_int_distribution<size_t> dAction(0, A[a]-1);
                        action_[a] = dAction(rnd_);
                    }
                }
            }
            // If nothing changed, there's no reason to do any more work.
            if (action_ == newAction_) continue;

            // Otherwise run LS on our new action.
            auto [newOptimizedAction, newOptimizedValue] = ls_(A, graph, newAction_);

            // If we have improved, store it as the new best.
            if (newOptimizedValue > 1.0) {
                action_ = newOptimizedAction;
                value_ = newOptimizedValue;
            }
        }

        return {action_, value_};
    }
}
