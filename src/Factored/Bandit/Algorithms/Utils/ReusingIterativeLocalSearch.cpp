#include <AIToolbox/Factored/Bandit/Algorithms/Utils/ReusingIterativeLocalSearch.hpp>

#include <AIToolbox/Seeder.hpp>
#include <AIToolbox/Utils/Probability.hpp>

namespace AIToolbox::Factored::Bandit {
    ReusingIterativeLocalSearch::ReusingIterativeLocalSearch(const double resetActionProbability, const double randomizeFactorProbability, const unsigned trialNum, const bool forceResetAction) :
        resetActionProbability_(resetActionProbability),
        randomizeFactorProbability_(randomizeFactorProbability),
        trialNum_(trialNum),
        forceResetAction_(forceResetAction),
        rnd_(Seeder::getSeed())
    {}

    ReusingIterativeLocalSearch::Result ReusingIterativeLocalSearch::operator()(const Action & A, const Graph & graph) {
        // If we haven't initialized the action yet, do so. Otherwise we keep
        // the old one, hoping that the graph has not changed too much and that
        // the new optimum is close to our old one.
        //
        // If needed we can also force a reset (in case we know that the new
        // graph has nothing in common with the old one).
        //
        // In both cases we need to recompute the action value, since the
        // values in the graph have likely changed (otherwise we wouldn't be
        // here).
        double value;
        if (forceResetAction_ || action_.empty())
            std::tie(action_, value) = ls_(A, graph);
        else
            value = LocalSearch::evaluateGraph(A, graph, action_);

        // In the trials we look around to see if we got stuck in some
        // local optima.
        for (unsigned i = 0; i < trialNum_; ++i) {
            // Either we completely start again from scratch...
            if (probabilityDistribution(rnd_) < resetActionProbability_) {
                newAction_ = makeRandomValue(A, rnd_);
            } else {
                newAction_ = action_;

                // Or we start from our current best action, and randomize
                // groups of agents. Note that we do not iterate over agents,
                // but on factors. This is because otherwise it would be much
                // more likely that LS simply climbs back to the same action as
                // before, ignoring our change.
                for (const auto & f : graph) {
                    if (probabilityDistribution(rnd_) >= randomizeFactorProbability_)
                        continue;

                    for (auto a : f.getVariables()) {
                        std::uniform_int_distribution<size_t> dAction(0, A[a]-1);
                        newAction_[a] = dAction(rnd_);
                    }
                }
            }
            // If nothing changed, there's no reason to do any more work.
            if (action_ == newAction_) continue;

            // Otherwise run LS on our new action.
            auto [newOptimizedAction, newOptimizedValue] = ls_(A, graph, newAction_);

            // If we have improved, store it as the new best.
            if (newOptimizedValue > value) {
                action_ = newOptimizedAction;
                value = newOptimizedValue;
            }
        }

        return {action_, value};
    }

    double ReusingIterativeLocalSearch::getResetActionProbability() const { return resetActionProbability_; }
    void ReusingIterativeLocalSearch::setResetActionProbability(double resetActionProbability) { resetActionProbability_ = resetActionProbability; }

    double ReusingIterativeLocalSearch::getRandomizeFactorProbability() const { return randomizeFactorProbability_; }
    void ReusingIterativeLocalSearch::setRandomizeFactorProbability(double randomizeFactorProbability) { randomizeFactorProbability_ = randomizeFactorProbability; }

    unsigned ReusingIterativeLocalSearch::getTrialNum() const { return trialNum_; }
    void ReusingIterativeLocalSearch::setTrialNum(unsigned trialNum) { trialNum_ = trialNum; }

    bool ReusingIterativeLocalSearch::getForceResetAction() const { return forceResetAction_; }
    void ReusingIterativeLocalSearch::setForceResetAction(bool forceResetAction) { forceResetAction_ = forceResetAction; }
}
